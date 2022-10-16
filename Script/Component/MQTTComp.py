import string
from dataclasses import dataclass

@dataclass
class MQTTComp():
    brokerIP : string
    brokerPort : string
    commandTopic : string
    connectStatus : bool
    createUDPTask : bool
