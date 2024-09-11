class NodeSubGraph:
    def __init__(self, config):
        self.altitude_threshold = None
        self.distance_threshold = None
        self.features_path = None
        self.use_altitude = None
        self.norm_adj_matrix = None
        self.node_attr = None
        self.weight_adj_matrix = None
        self.node_num = None
        self.nodes = None
        self.altitude = None
        self.pm25_path = None
        self.altitude_path = None
        self.city_path = None

        self.config = config
        if self.config.dataset_name == 'KnowAir':
            self.convert_KnowAir_to_subgraph()
        elif self.config.dataset_name == 'UrbanAir':
            self.convert_UrbanAir_to_subgraph()
        else:
            raise NotImplementedError

    def convert_KnowAir_to_subgraph(self):
        pass

    def convert_UrbanAir_to_subgraph(self):
        pass
