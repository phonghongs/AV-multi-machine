import string
from dataclasses import dataclass

@dataclass
class MQTTComp():
    brokerIP : string
    brokerPort : string
    commandTopic : string
    controlTopic : string
    timestampTopic : string
    timestampProcessTopic : string
    timestampValue : float
    timestampProcessValue : float
    connectStatus : bool
    createUDPTask : bool
