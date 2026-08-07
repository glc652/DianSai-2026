import struct
import unittest

from esp32_puzzle_link import (
    Delivery,
    EventType,
    Link,
    MODE_REQUEST_MESSAGE,
    PIECE_COUNT_MESSAGE,
    PIECE_DATA_MESSAGE,
    PIECE_FORMAT,
    ReliablePuzzleSender,
    SendResult,
    build_result_messages,
    cobs_encode,
    crc8,
    parse_mode_request,
)


class MemorySerial:
    def __init__(self):
        self.peer = None
        self.received = bytearray()
        self.transmitted = bytearray()

    @property
    def in_waiting(self):
        return len(self.received)

    def read(self, size=1):
        size = min(size, len(self.received))
        data = bytes(self.received[:size])
        del self.received[:size]
        return data

    def write(self, data):
        data = bytes(data)
        self.transmitted.extend(data)
        self.peer.received.extend(data)
        return len(data)


def serial_pair():
    left = MemorySerial()
    right = MemorySerial()
    left.peer = right
    right.peer = left
    return left, right


class PuzzleLinkTest(unittest.TestCase):
    def test_crc_reference(self):
        self.assertEqual(crc8(b"123456789"), 0xF4)

    def test_mode_2_wire_bytes(self):
        body = bytes((0x83, 0x00, 0x02))
        raw = body + bytes((crc8(body),))
        self.assertEqual(raw, bytes.fromhex("83 00 02 B8"))
        self.assertEqual(
            cobs_encode(raw) + b"\0",
            bytes.fromhex("02 83 03 02 B8 00"),
        )

    def test_result_payload_conversion(self):
        actions = [
            {
                "source_pickup_mm": [44.5, 42.5],
                "target_centroid_mm": [80.2, 190.4],
                "rotation_delta_deg": 35.26,
            },
            {
                "source_pickup_mm": [150.0, 89.0],
                "target_centroid_mm": [120.0, 210.0],
                "rotation_delta_deg": -45.0,
            },
        ]
        messages = build_result_messages(actions)
        self.assertEqual(messages[0], (PIECE_COUNT_MESSAGE, b"\x02"))
        self.assertEqual(
            struct.unpack(PIECE_FORMAT, messages[1][1]),
            (1, 45, 43, 80, 190, 3526),
        )
        self.assertEqual(
            struct.unpack(PIECE_FORMAT, messages[2][1]),
            (2, 150, 89, 120, 210, -4500),
        )

    def test_bidirectional_reliable_exchange(self):
        pi_serial, esp_serial = serial_pair()
        pi_link = Link(pi_serial)
        esp_link = Link(esp_serial)

        result = esp_link.send(
            MODE_REQUEST_MESSAGE,
            b"\x03",
            Delivery.RELIABLE,
        )
        self.assertEqual(result, SendResult.ACCEPTED)
        request = pi_link.poll()
        self.assertEqual(parse_mode_request(request), 3)
        delivered = esp_link.poll()
        self.assertEqual(delivered.type, EventType.DELIVERED)

        actions = [
            {
                "source_pickup_mm": [20.0, 30.0],
                "target_centroid_mm": [100.0, 200.0],
                "rotation_delta_deg": 30.0,
            },
            {
                "source_pickup_mm": [40.0, 50.0],
                "target_centroid_mm": [120.0, 220.0],
                "rotation_delta_deg": -60.0,
            },
        ]
        sender = ReliablePuzzleSender(pi_link)
        sender.start(actions)
        received = []
        for _ in range(20):
            pi_event = pi_link.poll()
            sender.step(pi_event)
            esp_event = esp_link.poll()
            if esp_event.type == EventType.MESSAGE:
                received.append(
                    (esp_event.message.type, esp_event.message.payload)
                )
            if sender.complete:
                break

        self.assertTrue(sender.complete)
        self.assertEqual(sender.delivered_count, 3)
        self.assertEqual(len(received), 3)
        self.assertEqual(received[0], (PIECE_COUNT_MESSAGE, b"\x02"))
        self.assertEqual(received[1][0], PIECE_DATA_MESSAGE)
        self.assertEqual(
            struct.unpack(PIECE_FORMAT, received[1][1]),
            (1, 20, 30, 100, 200, 3000),
        )
        self.assertEqual(
            struct.unpack(PIECE_FORMAT, received[2][1]),
            (2, 40, 50, 120, 220, -6000),
        )


if __name__ == "__main__":
    unittest.main()
