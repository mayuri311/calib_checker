"""Live ROS 2 node. Read-only on the driver topics; never touches recording or camera serial.

Subscribes : each session sensor's Image topic (best effort, depth 1) and optional FFC Bool topics.
Publishes  : /calib_checker/holds (std_msgs/String, JSON per hold) and /diagnostics.
Services   : /calib_checker/{reject_last, accept_last, skip_target, finish} (std_srvs/Trigger).
Params     : session (yaml path; empty -> configure in the web page), headless, viz_host,
             grpc_port, web_port, ui_port, output_dir.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Dict, Optional

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from .ros_io import image_to_numpy, stamp_ns
from .runtime import Runtime, _plain
from .session import Session, load_session

SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)


class _Worker(threading.Thread):
    """Holds only the newest message per sensor; detection never queues behind the driver."""

    def __init__(self, runtime: Runtime, sensor: str, logger):
        super().__init__(daemon=True, name=f"cc_{sensor}")
        self.runtime, self.sensor, self.logger = runtime, sensor, logger
        self.cv = threading.Condition()
        self.msg, self.t = None, 0.0
        self.running = True

    def put(self, msg) -> None:
        with self.cv:
            self.msg, self.t = msg, time.monotonic()
            self.cv.notify()

    def run(self) -> None:
        while self.running:
            with self.cv:
                while self.msg is None and self.running:
                    self.cv.wait(0.5)
                msg, t, self.msg = self.msg, self.t, None
            if msg is None:
                continue
            try:
                self.runtime.on_frame(self.sensor, image_to_numpy(msg), stamp_ns(msg.header), t)
            except Exception as e:  # keep the worker alive on a bad frame
                self.logger.error(f"{self.sensor}: {e}", throttle_duration_sec=5.0)


class CheckerNode(Node):
    def __init__(self):
        super().__init__("calib_checker")
        for name, default in (("session", ""), ("headless", False), ("viz_host", "localhost"),
                              ("grpc_port", 9876), ("web_port", 9090), ("ui_port", 8080), ("output_dir", "")):
            self.declare_parameter(name, default)
        self.runtime: Optional[Runtime] = None
        self.workers: Dict[str, _Worker] = {}
        self.pub_holds = self.create_publisher(String, "~/holds", 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        for srv in ("reject_last", "accept_last", "skip_target", "finish"):
            self.create_service(Trigger, f"~/{srv}", self._make_srv(srv))

    def p(self, name):
        return self.get_parameter(name).value

    # ------------------------------------------------------------------ lifecycle
    def start(self, session: Session) -> Runtime:
        if self.runtime is not None:
            return self.runtime
        from .viz_rerun import RerunViz

        def viz_factory(cams, T_ref, out_dir, thermal_like):
            return RerunViz(cams, T_ref, session.name, str(out_dir / "session.rrd"),
                            grpc_port=int(self.p("grpc_port")), web_port=int(self.p("web_port")),
                            host=self.p("viz_host"), thermal_like=thermal_like)

        self.runtime = Runtime(session, viz_factory, out_root=self.p("output_dir") or None, on_hold=self._publish_hold)
        for s in session.sensors:
            w = _Worker(self.runtime, s.name, self.get_logger())
            w.start()
            self.workers[s.name] = w
            self.create_subscription(Image, s.topic, w.put, SENSOR_QOS)
            if s.ffc_topic:
                self.create_subscription(Bool, s.ffc_topic,
                                         lambda m, n=s.name: self.runtime.on_ffc(n, bool(m.data)), 10)
        self.create_timer(0.1, lambda: self.runtime.tick())
        self.create_timer(1.0, self._publish_diag)
        self.get_logger().info(f"calib_checker running; output -> {self.runtime.out_dir}")
        return self.runtime

    def stop(self) -> None:
        for w in self.workers.values():
            w.running = False

    # ------------------------------------------------------------------ outputs
    def _publish_hold(self, rec) -> None:
        self.pub_holds.publish(String(data=json.dumps(_plain(rec.to_dict()))))

    def _publish_diag(self) -> None:
        st = self.runtime.status_dict()
        ds = DiagnosticStatus(name="calib_checker", hardware_id="calib_checker",
                              level=DiagnosticStatus.WARN if st["warnings"] else DiagnosticStatus.OK,
                              message=st["status"])
        flat = {"holds_accepted": st.get("holds_accepted", 0), "holds_total": st.get("holds_total", 0),
                "planned_remaining": st.get("planned_remaining", 0), "focus_edge": st.get("focus_edge"),
                **{f"coverage/{k}": v for k, v in st.get("coverage", {}).items()}}
        ds.values = [KeyValue(key=k, value=str(v)) for k, v in flat.items()]
        msg = DiagnosticArray(status=[ds])
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_diag.publish(msg)

    def _make_srv(self, name):
        def cb(_req, resp):
            if self.runtime is None:
                resp.success, resp.message = False, "not started"
                return resp
            resp.message = getattr(self.runtime, name)()
            resp.success = True
            return resp
        return cb


def main(args=None):
    rclpy.init(args=args)
    node = CheckerNode()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    try:
        if node.p("session"):
            node.start(load_session(node.p("session")))
        if node.p("headless"):
            if node.runtime is None:
                raise SystemExit("headless mode needs -p session:=<file>")
            while rclpy.ok():
                time.sleep(0.5)
        else:
            from .ui_web import run_ui
            run_ui(node, port=int(node.p("ui_port")))
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        if node.runtime is not None:
            node.runtime.finish()
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
