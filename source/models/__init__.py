from .transformer import GraphTransformer
from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN
from .fbnetgen import FBNETGEN
from .BNT import BrainNetworkTransformer
from .SNT import BrainSCNNTransformer

def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    if config.model.name == "SNT":
        return BNT(config).cuda()
    return eval(config.model.name)(config).cuda()
