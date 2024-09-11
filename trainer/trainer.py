from dataset.parser import Parser
from trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self._read_data()

    def _read_data(self):
        """
        construct train, valid, and test dataset
        :return: dataset loader
        """
        if self.config.dataset_name == 'KnowAir':
            self.train_dataset = Parser(config=self.config, mode='train')
            self.valid_dataset = Parser(config=self.config, mode='valid')
            self.test_dataset = Parser(config=self.config, mode='test')
            self.city_num = self.train_dataset.node_num
        elif self.config.dataset_name == 'UrbanAir':
            self.train_dataset = Parser(config=self.config, mode='train')
            self.valid_dataset = Parser(config=self.config, mode='valid')
            self.test_dataset = Parser(config=self.config, mode='test')
            self.city_num = self.train_dataset.node_num
        else:
            self.logger.error("Unsupported dataset type")
            raise ValueError('Unknown dataset')
