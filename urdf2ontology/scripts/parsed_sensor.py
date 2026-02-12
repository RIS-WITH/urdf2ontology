from dataclasses import dataclass, field
from typing import Optional, List, Dict

import xml.etree.ElementTree as ET
from typing import List


@dataclass
class ParsedSensor:
    link_name: str
    sensor_name: Optional[str]
    sensor_type: Optional[str]

    # topic: Optional[str]
    # update_rate: Optional[float]
    # always_on: Optional[bool]

    # Feature flags
    has_image: bool = False
    has_range: bool = False
    has_scan: bool = False
    has_imu: bool = False
    has_force_torque: bool = False
    has_contact: bool = False
    has_gps: bool = False

   # Extracted physical characteristics
    range_min: Optional[float] = None
    range_max: Optional[float] = None

    horizontal_fov: Optional[float] = None
    vertical_fov: Optional[float] = None

    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_format: Optional[str] = None

    scan_samples: Optional[int] = None
    scan_min_angle: Optional[float] = None
    scan_max_angle: Optional[float] = None

    # Raw parameters (fallback / debug)
    parameters: Dict[str, str] = field(default_factory=dict)
    
def parse_urdf_sensors(urdf_path: str) -> List[ParsedSensor]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    sensors: List[ParsedSensor] = []

    for gazebo in root.findall("gazebo"):
        reference = gazebo.attrib.get("reference")

        sensor_elem = gazebo.find("sensor")
        if sensor_elem is None:
            continue

        sensor_type = sensor_elem.attrib.get("type")
        sensor_name = sensor_elem.attrib.get("name")

        parsed = ParsedSensor(
            link_name=reference,
            sensor_name=sensor_name,
            sensor_type=sensor_type
            # topic=get_text(sensor_elem, "topic"),
            # update_rate=get_float(sensor_elem, "update_rate"),
            # always_on=get_bool(sensor_elem, "always_on"),
        )

        # ---------- Camera ----------
        camera = sensor_elem.find("camera")
        if camera is not None:
            parsed.has_image = True

            parsed.horizontal_fov = get_float(camera, "horizontal_fov")

            image = camera.find("image")
            if image is not None:
                parsed.image_width = get_int(image, "width")
                parsed.image_height = get_int(image, "height")
                parsed.image_format = get_text(image, "format")

            clip = camera.find("clip")
            if clip is not None:
                parsed.range_min = get_float(clip, "near")
                parsed.range_max = get_float(clip, "far")
                parsed.has_range = True

        # ---------- Ray / LiDAR ----------
        ray = sensor_elem.find("ray")
        if ray is not None:
            parsed.has_range = True
            parsed.has_scan = True

            range_elem = ray.find("range")
            if range_elem is not None:
                parsed.range_min = get_float(range_elem, "min")
                parsed.range_max = get_float(range_elem, "max")

            scan = ray.find("scan")
            if scan is not None:
                horizontal = scan.find("horizontal")
                if horizontal is not None:
                    parsed.scan_samples = get_int(horizontal, "samples")
                    parsed.scan_min_angle = get_float(horizontal, "min_angle")
                    parsed.scan_max_angle = get_float(horizontal, "max_angle")

        # ---------- IMU ----------
        if sensor_elem.find("imu") is not None:
            parsed.has_imu = True

        # ---------- Force / Torque ----------
        if sensor_elem.find("force_torque") is not None:
            parsed.has_force_torque = True

        # ---------- Contact ----------
        if sensor_elem.find("contact") is not None:
            parsed.has_contact = True

        # ---------- GPS ----------
        if sensor_elem.find("gps") is not None:
            parsed.has_gps = True

        # ---------- Capture raw parameters ----------
        for child in sensor_elem:
            if child.tag not in parsed.parameters:
                parsed.parameters[child.tag] = child.text.strip() if child.text else ""

        sensors.append(parsed)

    return sensors


# Helper functions
def get_text(parent, tag: str) -> Optional[str]:
    elem = parent.find(tag)
    return elem.text.strip() if elem is not None and elem.text else None


def get_float(parent, tag: str) -> Optional[float]:
    try:
        value = get_text(parent, tag)
        return float(value) if value is not None else None
    except ValueError:
        return None

def get_int(parent, tag: str):
    try:
        value = get_text(parent, tag)
        return int(value) if value is not None else None
    except ValueError:
        return None

def get_bool(parent, tag: str) -> Optional[bool]:
    value = get_text(parent, tag)
    if value is None:
        return None
    return value.lower() in ("true", "1", "yes")


if __name__ == "__main__":

    sensors = parse_urdf_sensors("/home/bdussard/ros2_ws/src/urdf2ontology/urdf/pr2.urdf")

    for s in sensors:
        print(s)