
# definitions could be fetched from the annotation in the ontology per type of component
definitions_dict = {
    'sensor': "A sensor is a component that perceives the environment.", 
    'end_effector' : "An end-effector is the terminal component of a kinematic chain that interacts physically with the environment.", 
    'actuator': "An actuator is a component that generates motion. In URDF models, actuators are often represented implicitly through actuated joints rather than standalone links. Links representing motors, transmissions, or servo housings should be classified as actuators.", 
    'body_part': "A body part is a structural component of the robot's morphology. These are supporting structural parts of the robot."}

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

