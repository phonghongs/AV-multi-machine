import configparser

class BackboneConfig():
    def __init__(self, configSession):
        self.error = False
        try:
            if 'videoSource' in configSession:
                self.videoSource = configSession['videoSource']
            if 'modelPath' in configSession:
                self.modelPath = configSession['modelPath']
            if 'serverIP' in configSession:
                self.serverIP = configSession['serverIP']
        except Exception as e:
            self.error = True
            print("[BackboneConfig] Error")

class ClietSegmentConfig():
    def __init__(self, configSession):
        self.error = False
        try:
            if 'videoSource' in configSession:
                self.videoSource = configSession['videoSource']
            if 'modelPath' in configSession:
                self.modelPath = configSession['modelPath']
            if 'serverIP' in configSession:
                self.serverIP = configSession['serverIP']
        except Exception as e:
            self.error = True
            print("[ClietSegmentConfig] Error")

class MQTTConfig():
    def __init__(self, configSession):
        self.error = False
        try:
            if 'brokerIP' in configSession:
                self.brokerIP = configSession['brokerIP']
            if 'brokerPort' in configSession:
                self.brokerPort = configSession['brokerPort']
            if 'mqttTopic' in configSession:
                self.mqttTopic = configSession['mqttTopic']
            if 'controlTopic' in configSession:
                self.controlTopic = configSession['controlTopic']
        except Exception as e:
            self.error = True
            print("[MQTTConfig] Error")

class SerialConfig():
    def __init__(self, configSession):
        self.error = False
        try:
            if 'serialPort' in configSession:
                self.serialPort = configSession['serialPort']
            if 'seralBaudraet' in configSession:
                self.seralBaudraet = configSession['seralBaudraet']
        except Exception as e:
            self.error = True
            print("[SerialConfig] Error")


class PareSystemConfig():
    def __init__(self, configPath):
        self._config = configparser.ConfigParser()
        self._config.read(configPath)
        self.isHaveConfig = True
        self.clientSegmentCfg = []
        self.mqttCfg = []
        self.serialCfg = []
        
        if (self._config == []):
            self.isHaveConfig = False
        
        try:
            self.backboneCfg = BackboneConfig(self._config['backbone'])
            self.clientSegmentCfg = ClietSegmentConfig(self._config['clientsegment'])
            self.mqttCfg = MQTTConfig(self._config['mqtt'])
            self.serialCfg = SerialConfig(self._config['serial'])
        
            if (
                self.backboneCfg.error or 
                self.clientSegmentCfg.error or
                self.mqttCfg.error or
                self.serialCfg.error
                ):
                self.isHaveConfig = False
        except Exception as e:
            self.isHaveConfig = False
            print("[Config] Error when pare config")

        