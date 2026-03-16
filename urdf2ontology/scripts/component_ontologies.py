
# definitions could be fetched from the annotation in the ontology per type of component
definitions_dict = {
    'sensor': "A sensor is a component that perceives the environment.", 
    # 'end_effector' : "An end-effector is the terminal component of a kinematic chain that interacts physically with the environment.",
    'end_effector' : "An end effector or tool head is the device at the end of a robotic arm, designed to interact physically with the environment.", 
    'actuator': "An actuator is a component that generates motion. In URDF models, actuators are often represented implicitly through actuated joints rather than standalone links. Links representing motors, transmissions, or servo housings should be classified as actuators.", 
    'body_part': "A body part is a structural component of the robot's morphology."}

# ontologies should be fetched per component task
end_effector_ontology = {
    "EndEffector": {
        "Gripper": {
            "TwoFingerGripper": {},
            "MultiFingerGripper": {},
            "AdaptiveGripper": {},
            "SoftRoboticGripper": {},
            "VacuumGripper": {},
            "MagneticGripper": {}
        },
        "Tool": {
            "WeldingTool": {
                "ArcWelder": {},
                "SpotWelder": {},
                "LaserWelder": {}
            },
            "PaintingTool": {
                "SprayGun": {}
            },
            "ScrewFasteningTool": {
                "Screwdriver": {}
            },
            "DrillingTool": {
                "Drill": {}
            },
            "DeburringTool": {},
            "CuttingTool": {}
        },
        "SensorsAsEndEffector": {
            "ForceTorqueSensor": {},
            "InspectionProbe": {}
        },
        "HybridTransportEndEffector": {
            "ConveyorAttachment": {},
            "TrayAttachment": {}
        },
        "UnknownEndEffector":{}
    }
}

actuator_ontology = {
    "Actuator": {
        "RotaryActuator": {
            "ElectricMotor": {},
            "ServoMotor": {},
            "StepperMotor": {},
            "HydraulicRotaryActuator": {},
            "PneumaticRotaryActuator": {}
        },
        "LinearActuator": {
            "ElectricLinearActuator": {},
            "HydraulicLinearActuator": {},
            "PneumaticLinearActuator": {},
            "LeadScrewActuator": {},
            "RackAndPinionActuator": {}
        },
        "SpecializedActuator": {
            "ShapeMemoryAlloy": {},
            "PiezoelectricActuator": {},
            "MagneticLevitationActuator": {}
        },
        "GripperActuator": {
            "ParallelJawActuator": {},
            "UnderactuatedFingerActuator": {},
            "SoftGripperActuator": {},
            "VacuumGripperActuator": {}
        },
        "JointActuator": {
            "RevoluteJointActuator": {},
            "PrismaticJointActuator": {},
            "SphericalJointActuator": {}
        },
        "UnknownActuator":{}
    }
}

sensor_ontology = {
    "Sensor": {
        "ProximitySensor": {
            "InfraredSensor": {},
            "UltrasonicSensor": {},
            "CapacitiveSensor": {},
            "InductiveSensor": {}
        },
        "VisionSensor": {
            "Camera": {
                "RGBCamera": {},
                "DepthCamera": {},
                "StereoCamera": {},
                "RGBDCamera": {},
                "InfraredCamera": {}
            },
            "Lidar": {},
            "TimeOfFlightCamera": {}
        },
        "ForceTorqueSensor": {
            "SixAxisForceTorqueSensor": {},
            "SingleAxisForceSensor": {}
        },
        "InertialSensor": {
            "IMU": {},
            "Gyroscope": {},
            "Accelerometer": {}
        },
        "TouchAndTactileSensor": {
            "TactileArraySensor": {},
            "PressureSensor": {},
            "ForceSensitiveResistor": {}
        },
        "EnvironmentalSensor": {
            "TemperatureSensor": {},
            "HumiditySensor": {},
            "GasSensor": {}
        },
        "PositionAndMotionSensor": {
            "Encoder": {},
            "Potentiometer": {},
            "MagneticEncoder": {}
        },
        "AuditorySensor": {
            "Microphone": {}
        },
        "UnknownSensor":{}
    }
}

bodypart_ontology = {
    "BodyPart": {
        "Base": {
            "FixedBase": {},
            "MobileBase": {}
        },
        "Arm": {
            "Shoulder": {},
            "UpperArm": {},
            "Elbow": {},
            "Forearm": {},
            "Wrist": {}
        },
        "Leg": {
            "Hip": {},
            "Thigh": {},
            "Knee": {},
            "Shank": {},
            "Ankle": {},
            "Foot": {}
        },
        "Torso": {
            "Chest": {},
            "Abdomen": {},
            "Pelvis": {}
        },
        "Head": {
            "Neck": {},
            "HeadMain": {}
        },
        "HandsAndFeet": {
            "Hand": {},
            "Fingers": {},
            "Thumb": {},
            "Toe": {},
            "FootMain": {}
        },
        "OtherStructuralLink": {
            "Frame": {},
            "Support": {},
            "Connector": {},
            "Bracket": {}
        },
        "UnknownBodyPart":{}
    }
}

# ================ V2 ============

communication_ontology = {
    "CommunicationDevice": {
        "AudioInputDevice": {
            "Microphone": {}
        },
        "AudioOutputDevice": {
            "Speaker": {}
        },
        "VisualOutputDevice": {
            "Display": {},
            "LEDIndicator": {},
            "Projector": {}
        },
        "HapticOutputDevice": {
            "VibrationMotor": {}
        },
        "WirelessCommunicationModule": {
            "WiFiModule": {},
            "BluetoothModule": {},
            "RFModule": {}
        },
        "WiredCommunicationInterface": {
            "EthernetPort": {},
            "USBInterface": {},
            "SerialInterface": {}
        },
        "UnknownCommunicationDevice": {}
    }
}

structural_ontology = {
    "StructuralComponent": {
        "Base": {
            "FixedBase": {},
            "MobileBase": {}
        },
        "AnatomicalSegment": {
            "Torso": {},
            "Shoulder": {},
            "UpperArm": {},
            "Elbow": {},
            "Forearm": {},
            "Wrist": {},
            "Hip": {},
            "Thigh": {},
            "Knee": {},
            "Shank": {},
            "Ankle": {},
            "Neck": {},
            "Head": {}
        },
        "TerminalStructure": {
            "Hand": {},
            "Foot": {},
            "Finger": {},
            "Thumb": {},
            "Toe": {}
        },
        "SupportStructure": {
            "Frame": {},
            "Bracket": {},
            "Connector": {},
            "Mount": {},
            "Housing": {}
        },
        "UnknownStructuralComponent": {}
    }
}

end_effector_ontology = {
    "EndEffector": {
        "Gripper": {
            "TwoFingerGripper": {},
            "MultiFingerGripper": {},
            "AdaptiveGripper": {},
            "SoftRoboticGripper": {},
            "VacuumGripper": {},
            "MagneticGripper": {}
        },
        "Tool": {
            "WeldingTool": {
                "ArcWelder": {},
                "SpotWelder": {},
                "LaserWelder": {}
            },
            "CuttingTool": {},
            "DrillingTool": {
                "Drill": {}
            },
            "ScrewFasteningTool": {
                "Screwdriver": {}
            },
            "PaintingTool": {
                "SprayGun": {}
            },
            "PolishingTool": {},
            "DeburringTool": {},
            "SamplingTool": {},
            "InspectionTool": {}
        },
        "MaterialHandlingAttachment": {
            "TrayAttachment": {},
            "ConveyorAttachment": {}
        },
        "UnknownEndEffector": {}
    }
}

sensor_ontology = {
    "Sensor": {
        "VisionSensor": {
            "RGBCamera": {},
            "DepthCamera": {},
            "StereoCamera": {},
            "RGBDCamera": {},
            "InfraredCamera": {},
            "Lidar": {},
            "TimeOfFlightCamera": {}
        },
        "ForceTorqueSensor": {
            "SixAxisForceTorqueSensor": {},
            "SingleAxisForceSensor": {}
        },
        "InertialSensor": {
            "IMU": {},
            "Gyroscope": {},
            "Accelerometer": {}
        },
        "ProximitySensor": {
            "InfraredSensor": {},
            "UltrasonicSensor": {},
            "CapacitiveSensor": {},
            "InductiveSensor": {}
        },
        "TactileSensor": {
            "TactileArraySensor": {},
            "PressureSensor": {},
            "ForceSensitiveResistor": {}
        },
        "PositionSensor": {
            "Encoder": {},
            "MagneticEncoder": {},
            "Potentiometer": {}
        },
        "EnvironmentalSensor": {
            "TemperatureSensor": {},
            "HumiditySensor": {},
            "GasSensor": {}
        },
        "AuditorySensor": {
            "Microphone": {}
        },
        "UnknownSensor": {}
    }
}

actuator_ontology = {
    "Actuator": {
        "RotaryActuator": {
            "ElectricMotor": {},
            "ServoMotor": {},
            "StepperMotor": {},
            "HydraulicRotaryActuator": {},
            "PneumaticRotaryActuator": {}
        },
        "LinearActuator": {
            "ElectricLinearActuator": {},
            "HydraulicLinearActuator": {},
            "PneumaticLinearActuator": {},
            "LeadScrewActuator": {},
            "RackAndPinionActuator": {}
        },
        "ElasticActuator": {
            "SeriesElasticActuator": {},
            "VariableStiffnessActuator": {}
        },
        "SpecializedActuator": {
            "PiezoelectricActuator": {},
            "ShapeMemoryAlloyActuator": {},
            "MagneticLevitationActuator": {}
        },
        "JointActuator": {
            "RevoluteJointActuator": {},
            "PrismaticJointActuator": {},
            "SphericalJointActuator": {}
        },
        "UnknownActuator": {}
    }
}