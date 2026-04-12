from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.lib import hub
import psutil
import os
import time
import json
import logging
from webob import Response

logger = logging.getLogger(__name__)

CONTROLLER_ID = 1
WSGI_PORT = 8080
OFP_PORT = 6633

RYU_APP_INSTANCE_NAME = "ryu_app_c1_instance" # Tên instance dùng để chia sẻ giữa Ryu app và REST API

class RyuAppC1(app_manager.RyuApp):
    """
    Ryu App cho Controller.
    Implements L2 learning switch + packet-in monitoring.
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(RyuAppC1, self).__init__(*args, **kwargs)

        # MAC learning table: {dpid: {mac: port}}
        self.mac_to_port = {}

        # Metrics để expose ra REST API và Prometheus
        self.packet_in_count = 0          # tổng số packet-in nhận được
        self.packet_in_rate = 0.0         # packet-in/giây (tính mỗi 5s)
        self.connected_switches = set()   # dpid của switches đang kết nối

        # psutil: lấy process hiện tại để đo CPU/RAM
        self._process = psutil.Process(os.getpid())
        self._last_pkt_count = 0
        self._last_measure_time = time.time()

        # Đăng ký instance vào WSGI để REST API có thể truy cập
        wsgi = kwargs["wsgi"]
        wsgi.register(MonitorController, {RYU_APP_INSTANCE_NAME: self})

        # Background thread đo metrics mỗi 5 giây
        self._monitor_thread = hub.spawn(self._monitor_loop)

        logger.info(f"[Controller {CONTROLLER_ID}] Khởi động tại OFP:{OFP_PORT} REST:{WSGI_PORT}")

    # ------------------------------------------------------------------
    # OpenFlow event handlers
    # ------------------------------------------------------------------

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Khi switch kết nối: cài table-miss flow để gửi packet lên controller."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.connected_switches.add(datapath.id)
        logger.info(f"[C{CONTROLLER_ID}] Switch kết nối: dpid={datapath.id:#x} | "
                    f"Tổng: {len(self.connected_switches)} switches")

        # Table-miss: gửi tất cả packet không match lên controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, priority = 0, match = match, actions = actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Xử lý packet-in: học MAC và cài flow rule."""
        self.packet_in_count += 1

        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # Bỏ qua LLDP
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        # Nếu biết MAC đích → forward; không → flood
        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)

        actions = [parser.OFPActionOutput(out_port)]

        # Cài flow rule nếu không flood
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self._add_flow(datapath, priority=1, match = match, actions = actions, buffer_id = msg.buffer_id)
                return
            else:
                self._add_flow(datapath, priority = 1, match = match, actions = actions)

        # Gửi packet ra port
        data = None if msg.buffer_id != ofproto.OFP_NO_BUFFER else msg.data
        out = parser.OFPPacketOut(datapath = datapath, buffer_id = msg.buffer_id, in_port = in_port, actions = actions, data = data)
        datapath.send_msg(out)

    def _add_flow(self, datapath, priority, match, actions, buffer_id=None):
        """Helper: thêm flow entry vào switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        kwargs = dict(datapath=datapath, priority=priority, match=match, instructions=inst)
        if buffer_id and buffer_id != ofproto.OFP_NO_BUFFER:
            kwargs["buffer_id"] = buffer_id

        mod = parser.OFPFlowMod(**kwargs)
        datapath.send_msg(mod)

    # ------------------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------------------

    def _monitor_loop(self):
        """Background thread: đo packet-in rate mỗi 5 giây."""
        while True:
            hub.sleep(5)
            now = time.time()
            elapsed = now - self._last_measure_time
            if elapsed > 0:
                delta_pkt = self.packet_in_count - self._last_pkt_count
                self.packet_in_rate = delta_pkt / elapsed
                self._last_pkt_count = self.packet_in_count
                self._last_measure_time = now

    def get_metrics(self) -> dict:
        """
        Trả về metrics hiện tại của controller này.
        Được gọi bởi REST API endpoint /monitor/load.
        Format khớp với sdn_env.py _get_state_real().
        """
        cpu = self._process.cpu_percent(interval = None) / 100.0   # normalize [0,1]
        mem = self._process.memory_percent() / 100.0              # normalize [0,1]
        pkt_rate_norm = min(self.packet_in_rate / 1000.0, 1.0)    # normalize, cap ở 1000 pkt/s

        return {
            "controller_id": CONTROLLER_ID,
            "cpu": round(cpu, 4),
            "ram": round(mem, 4),
            "packet_in_rate": round(pkt_rate_norm, 4),
            "packet_in_total": self.packet_in_count,
            "switch_count": len(self.connected_switches),
            "connected_switches": [hex(dpid) for dpid in self.connected_switches],
        }

# ------------------------------------------------------------------
# REST API Controller (dùng chung monitor_api.py pattern)
# ------------------------------------------------------------------

class MonitorController(ControllerBase):
    """REST API endpoints cho Controller 1."""

    def __init__(self, req, link, data, **config):
        super(MonitorController, self).__init__(req, link, data, **config)
        self.app: RyuAppC1 = data[RYU_APP_INSTANCE_NAME]

    @route("monitor", "/monitor/load", methods=["GET"])
    def get_load(self, req, **kwargs):
        """
        GET /monitor/load
        Trả về metrics CPU/RAM/packet_in_rate cho RL agent và Prometheus.
        """
        metrics = self.app.get_metrics()
        return Response(content_type = "application/json", body = json.dumps(metrics))

    @route("monitor", "/monitor/switches", methods=["GET"])
    def get_switches(self, req, **kwargs):
        """GET /monitor/switches — Danh sách switches đang kết nối."""
        data = {
            "controller_id": CONTROLLER_ID,
            "switch_count": len(self.app.connected_switches),
            "switches": [hex(dpid) for dpid in self.app.connected_switches],
        }
        return Response(content_type = "application/json", body = json.dumps(data))