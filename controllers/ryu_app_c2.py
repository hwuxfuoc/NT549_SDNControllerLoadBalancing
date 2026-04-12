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

CONTROLLER_ID = 2
WSGI_PORT = 8081
OFP_PORT = 6634

RYU_APP_INSTANCE_NAME = "ryu_app_c2_instance"

class RyuAppC2(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(RyuAppC2, self).__init__(*args, **kwargs)

        self.mac_to_port = {}
        
        self.packet_in_count = 0
        self.packet_in_rate = 0.0
        self.connected_switches = set()

        self._process = psutil.Process(os.getpid())
        self._last_pkt_count = 0
        self._last_measure_time = time.time()

        wsgi = kwargs["wsgi"]
        wsgi.register(MonitorController, {RYU_APP_INSTANCE_NAME: self})

        self._monitor_thread = hub.spawn(self._monitor_loop)
        logger.info(f"[Controller {CONTROLLER_ID}] Khởi động tại OFP:{OFP_PORT} REST:{WSGI_PORT}")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.connected_switches.add(datapath.id)
        logger.info(f"[C{CONTROLLER_ID}] Switch kết nối: dpid={datapath.id:#x} | "
                    f"Tổng: {len(self.connected_switches)} switches")

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, priority = 0, match = match, actions = actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        self.packet_in_count += 1

        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self._add_flow(datapath, priority = 1, match = match, actions = actions, buffer_id = msg.buffer_id)
                return
            else:
                self._add_flow(datapath, priority = 1, match = match, actions = actions)

        data = None if msg.buffer_id != ofproto.OFP_NO_BUFFER else msg.data
        out = parser.OFPPacketOut(datapath = datapath, buffer_id = msg.buffer_id, in_port = in_port, actions = actions, data=data)
        datapath.send_msg(out)

    def _add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        kwargs = dict(datapath=datapath, priority=priority, match=match, instructions=inst)
        if buffer_id and buffer_id != ofproto.OFP_NO_BUFFER:
            kwargs["buffer_id"] = buffer_id
        datapath.send_msg(parser.OFPFlowMod(**kwargs))

    def _monitor_loop(self):
        while True:
            hub.sleep(5)
            now = time.time()
            elapsed = now - self._last_measure_time
            if elapsed > 0:
                self.packet_in_rate = (self.packet_in_count - self._last_pkt_count) / elapsed
                self._last_pkt_count = self.packet_in_count
                self._last_measure_time = now

    def get_metrics(self) -> dict:
        cpu = self._process.cpu_percent(interval = None) / 100.0
        mem = self._process.memory_percent() / 100.0
        pkt_rate_norm = min(self.packet_in_rate / 1000.0, 1.0)
        return {
            "controller_id": CONTROLLER_ID,
            "cpu": round(cpu, 4),
            "ram": round(mem, 4),
            "packet_in_rate": round(pkt_rate_norm, 4),
            "packet_in_total": self.packet_in_count,
            "switch_count": len(self.connected_switches),
            "connected_switches": [hex(dpid) for dpid in self.connected_switches],
        }

class MonitorController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(MonitorController, self).__init__(req, link, data, **config)
        self.app: RyuAppC2 = data[RYU_APP_INSTANCE_NAME]

    @route("monitor", "/monitor/load", methods=["GET"])
    def get_load(self, req, **kwargs):
        return Response(content_type="application/json", body = json.dumps(self.app.get_metrics()))

    @route("monitor", "/monitor/switches", methods=["GET"])
    def get_switches(self, req, **kwargs):
        data = {
            "controller_id": CONTROLLER_ID,
            "switch_count": len(self.app.connected_switches),
            "switches": [hex(dpid) for dpid in self.app.connected_switches],
        }
        return Response(content_type = "application/json", body = json.dumps(data))