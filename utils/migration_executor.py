import socket
import struct
import logging
import time
from typing import Dict, List, Optional, Tuple

from utils.api_client import RyuAPIClient

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# OpenFlow 1.3 constants cho Role Request
# https://opennetworking.org/wp-content/uploads/2014/10/openflow-spec-v1.3.0.pdf
# ------------------------------------------------------------------

OFP_VERSION = 0x04          # OpenFlow 1.3
OFP_TYPE_ROLE_REQUEST = 24  # OFPT_ROLE_REQUEST
OFP_ROLE_NOCHANGE = 0
OFP_ROLE_EQUAL = 1
OFP_ROLE_MASTER = 2
OFP_ROLE_SLAVE = 3

# Header: version(1) + type(1) + length(2) + xid(4) = 8 bytes
# Role body: role(4) + pad(4) + generation_id(8) = 16 bytes
OFP_ROLE_REQUEST_SIZE = 24


def _build_role_request(role: int, generation_id: int = 0, xid: int = 1) -> bytes:
    """
    Tạo OpenFlow 1.3 Role Request message (bytes).

    Args:
        role: OFP_ROLE_MASTER hoặc OFP_ROLE_SLAVE
        generation_id: monotonically increasing ID (dùng 0 cho đơn giản)
        xid: transaction ID

    Returns:
        bytes — raw OpenFlow message
    """
    header = struct.pack("!BBHI", OFP_VERSION, OFP_TYPE_ROLE_REQUEST, OFP_ROLE_REQUEST_SIZE, xid)
    body = struct.pack("!IIQ", role, 0, generation_id)  # role, pad, generation_id
    return header + body


class MigrationExecutor:
    """
    Thực thi switch migration trong SDN cluster.

    Hỗ trợ 2 chế độ:
    - use_mock=True: cập nhật switch_assignment nội bộ (cho training)
    - use_mock=False: gửi OpenFlow Role Request thực (cho deployment)
    """

    def __init__(self, num_controllers: int = 3, num_switches: int = 12, use_mock: bool = True, api_client: Optional[RyuAPIClient] = None, ofp_timeout: float = 2.0):
        self.num_controllers = num_controllers
        self.num_switches = num_switches
        self.use_mock = use_mock
        self.ofp_timeout = ofp_timeout

        # api_client dùng để lấy thông tin switch và controller
        self.api_client = api_client or RyuAPIClient()

        # switch_assignment[switch_id] = controller_id (0-indexed)
        # Khởi tạo phân phối đều
        import numpy as np
        self.switch_assignment = np.array([i % num_controllers for i in range(num_switches)])

        # Lịch sử migration để debug và tính metrics
        self.migration_log: List[Dict] = []

        # Controller OFP configs: {controller_id (1-indexed) → (host, ofp_port)}
        self._ofp_endpoints: Dict[int, Tuple[str, int]] = {cfg["id"]: (cfg["host"], cfg["ofp_port"]) for cfg in self.api_client.configs}

        logger.info(
            f"[MigrationExecutor] mode={'MOCK' if use_mock else 'REAL'} | "
            f"{num_controllers} controllers | {num_switches} switches"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def migrate(self, switch_id: int, target_controller_id: int) -> bool:
        """
        Migrate switch từ controller hiện tại sang target_controller.

        Args:
            switch_id: index switch (0-indexed, tương ứng s1=0, s2=1,...)
            target_controller_id: controller_id đích (1-indexed, khớp với Ryu app)

        Returns:
            True nếu migration thành công.
        """
        if not self._validate(switch_id, target_controller_id):
            return False

        # Lấy controller hiện tại (1-indexed)
        current_ctrl_idx = int(self.switch_assignment[switch_id])  # 0-indexed
        current_controller_id = current_ctrl_idx + 1               # 1-indexed

        if current_controller_id == target_controller_id:
            logger.debug(f"[Migration] s{switch_id + 1} đã ở C{target_controller_id}, bỏ qua")
            return False

        logger.info(f"[Migration] s{switch_id + 1}: C{current_controller_id} → C{target_controller_id}")

        success = False
        if self.use_mock:
            success = self._migrate_mock(switch_id, target_controller_id)
        else:
            success = self._migrate_real(switch_id, current_controller_id, target_controller_id)

        if success:
            # Cập nhật assignment (0-indexed)
            self.switch_assignment[switch_id] = target_controller_id - 1
            self.migration_log.append({
                "time": time.time(),
                "switch_id": switch_id,
                "switch_name": f"s{switch_id + 1}",
                "from_controller": current_controller_id,
                "to_controller": target_controller_id,
            })
            logger.info(f"[Migration] ✓ s{switch_id + 1} → C{target_controller_id} thành công")

        else:
            logger.warning(f"[Migration] ✗ s{switch_id + 1} → C{target_controller_id} thất bại")

        return success

    def migrate_by_name(self, switch_name: str, target_controller_id: int) -> bool:
        """
        Migrate switch theo tên (ví dụ: "s3" → switch_id=2).

        Args:
            switch_name: tên switch dạng "sN" (N bắt đầu từ 1)
            target_controller_id: controller_id đích (1-indexed)
        """
        if not switch_name.startswith("s"):
            logger.error(f"[Migration] switch_name không hợp lệ: {switch_name}")
            return False
        
        try:
            switch_id = int(switch_name[1:]) - 1  # "s3" → index 2

        except ValueError:
            logger.error(f"[Migration] Không parse được switch_name: {switch_name}")
            return False
        
        return self.migrate(switch_id, target_controller_id)

    def get_migration_count(self) -> int:
        return len(self.migration_log)

    def get_switch_assignment(self) -> Dict[str, int]:
        """Trả về assignment dạng {switch_name: controller_id (1-indexed)}."""
        return {f"s{i + 1}": int(self.switch_assignment[i]) + 1 for i in range(self.num_switches)}

    def print_assignment(self) -> None:
        """In trạng thái assignment ra console để debug."""
        import numpy as np
        print("\n=== Switch Assignment ===")
        for ctrl_id in range(1, self.num_controllers + 1):
            switches = [f"s{i+1}" for i in range(self.num_switches) if int(self.switch_assignment[i]) + 1 == ctrl_id]
            print(f"  Controller {ctrl_id}: {switches} ({len(switches)} switches)")
        counts = [int(np.sum(self.switch_assignment == i)) for i in range(self.num_controllers)]
        print(f"  Variance(count): {np.var(counts):.4f}")

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _validate(self, switch_id: int, target_controller_id: int) -> bool:
        if not (0 <= switch_id < self.num_switches):
            logger.error(f"[Migration] switch_id={switch_id} ngoài phạm vi [0, {self.num_switches})")
            return False
        valid_ctrl_ids = [cfg["id"] for cfg in self.api_client.configs]

        if target_controller_id not in valid_ctrl_ids:
            logger.error(f"[Migration] target_controller_id={target_controller_id} không hợp lệ")
            return False
        
        return True

    def _migrate_mock(self, switch_id: int, target_controller_id: int) -> bool:
        """Mock migration: chỉ cập nhật assignment nội bộ."""
        # Giả lập độ trễ nhỏ của migration thực
        time.sleep(0.01)
        return True

    def _migrate_real(self, switch_id: int, current_controller_id: int, target_controller_id: int) -> bool:
        """
        Thực thi migration thực bằng OpenFlow Role Request.

        Bước 1: Gửi SLAVE đến controller hiện tại.
        Bước 2: Gửi MASTER đến controller đích.

        Lưu ý: Switch phải đã kết nối với CẢ HAI controllers
        (multi-controller mode trong Mininet custom_topo.py).
        """
        # Lấy dpid của switch từ Ryu API
        switch_dpid = self._get_switch_dpid(switch_id, current_controller_id)
        if switch_dpid is None:
            logger.error(
                f"[Migration] Không tìm được dpid của s{switch_id+1} "
                f"từ controller {current_controller_id}"
            )
            return False

        # Bước 1: Demote controller hiện tại xuống SLAVE
        step1 = self._send_role_request(controller_id = current_controller_id, dpid = switch_dpid, role = OFP_ROLE_SLAVE)

        if not step1:
            logger.error(f"[Migration] Không gửi được SLAVE đến C{current_controller_id}")
            return False

        # Bước 2: Promote controller đích lên MASTER
        step2 = self._send_role_request(controller_id = target_controller_id, dpid = switch_dpid, role = OFP_ROLE_MASTER)

        if not step2:
            # Rollback: khôi phục MASTER về controller cũ
            self._send_role_request(current_controller_id, switch_dpid, OFP_ROLE_MASTER)
            logger.error(f"[Migration] Không gửi được MASTER đến C{target_controller_id} — đã rollback")
            return False

        return True

    def _get_switch_dpid(self, switch_id: int, controller_id: int) -> Optional[int]:
        """
        Lấy dpid (int) của switch từ Ryu REST API.
        switch_id là 0-indexed → dpid thường = switch_id + 1 trong Mininet.
        """
        switches_hex = self.api_client.get_switch_list(controller_id)
        if not switches_hex:
            # Fallback: Mininet đặt dpid = switch number (s1 → dpid=1)
            return switch_id + 1

        # Tìm switch có dpid tương ứng (s1=dpid:1, s2=dpid:2, ...)
        expected_dpid = switch_id + 1
        for hex_dpid in switches_hex:
            try:
                dpid = int(hex_dpid, 16)
                if dpid == expected_dpid:
                    return dpid
            except ValueError:
                continue

        logger.warning(
            f"[Migration] Không tìm thấy dpid={expected_dpid} trong switch list "
            f"của C{controller_id}: {switches_hex}"
        )

        return None

    def _send_role_request(self, controller_id: int, dpid: int, role: int) -> bool:
        """
        Gửi OpenFlow Role Request trực tiếp đến switch qua TCP socket.

        Lưu ý: Trong thực tế, Role Request được gửi từ controller đến switch,
        không phải từ executor. Cách đúng hơn là gọi API của Ryu để Ryu gửi.
        Implement này dùng cho mục đích demo/testing.
        """
        endpoint = self._ofp_endpoints.get(controller_id)

        if endpoint is None:
            logger.error(f"[OFP] Không tìm thấy OFP endpoint cho C{controller_id}")
            return False

        host, port = endpoint
        role_name = "MASTER" if role == OFP_ROLE_MASTER else "SLAVE"

        try:
            msg = _build_role_request(role=role, generation_id=int(time.time()))
            with socket.create_connection((host, port), timeout=self.ofp_timeout) as sock:
                sock.sendall(msg)
            logger.debug(
                f"[OFP] Gửi {role_name} đến C{controller_id} ({host}:{port}) "
                f"cho switch dpid = {dpid}"
            )
            return True
        
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            logger.error(f"[OFP] Lỗi kết nối C{controller_id} ({host}:{port}): {e}")
            return False


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="SDN Switch Migration Tool")
    parser.add_argument("--switch", type=int, required=True, help="Switch ID (1-indexed, ví dụ: 3 cho s3)")
    parser.add_argument("--target-controller", type=int, required=True, help="Controller đích (1, 2, hoặc 3)")
    parser.add_argument("--mock", action="store_true", default=True, help="Dùng mock mode (default)")
    parser.add_argument("--real", action="store_true", help="Dùng OpenFlow thực")
    args = parser.parse_args()

    use_mock = not args.real

    executor = MigrationExecutor(num_controllers = 3, num_switches = 12, use_mock = use_mock)

    print("\n=== Trước migration ===")
    executor.print_assignment()

    switch_id = args.switch - 1  # chuyển từ 1-indexed sang 0-indexed
    success = executor.migrate(switch_id, args.target_controller)

    print(f"\nKết quả: {'✓ Thành công' if success else '✗ Thất bại'}")
    print("\n=== Sau migration ===")
    executor.print_assignment()