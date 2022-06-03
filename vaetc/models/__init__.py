from .abstract import RLModel, AutoEncoderRLModel, GaussianEncoderAutoEncoderRLModel

from .byname import model_by_params, register_model

from .cnn import CNNClassifier
register_model("cnn", CNNClassifier)
from .ae import AutoEncoder
register_model("ae", AutoEncoder)
from .aae import AAE
register_model("aae", AAE)
from .vae import VAE
register_model("vae", VAE)
from .cvae import CVAE
register_model("cvae", CVAE)
from .bvae import BetaVAE
register_model("bvae", BetaVAE)
from .cabvae import CyclicalAnnealingBetaVAE
register_model("cabvae", CyclicalAnnealingBetaVAE)
from .annealedvae import AnnealedVAE
register_model("annealedvae", AnnealedVAE)
from .btcvae import BetaTCVAE
register_model("btcvae", BetaTCVAE)
from .dipvae import DIPVAEI, DIPVAEII
register_model("dipvaei", DIPVAEI)
register_model("dipvaeii", DIPVAEII)
from .infovae import MMDVAE
register_model("infovae", MMDVAE)
register_model("wae", MMDVAE)
from .factorvae import FactorVAE
register_model("factorvae", FactorVAE)

from .vitae import VITAE
register_model("vitae", VITAE)
from .sigmavae import SigmaVAE
register_model("sigmavae", SigmaVAE)
from .dfcvae import DFCVAE
register_model("dfcvae", DFCVAE)
from .vladderae import VariationalLadderAutoEncoders
register_model("vladderae", VariationalLadderAutoEncoders)
from .laddervae import LadderVAE
register_model("laddervae", LadderVAE)
from .guidedvae import GuidedVAE
register_model("guidedvae", GuidedVAE)
from .bavae import BetaAnnealedVAE
register_model("bavae", BetaAnnealedVAE)

from .vaegan import VAEGAN
register_model("vaegan", VAEGAN)
from .introvae import IntroVAE
register_model("introvae", IntroVAE)
from .sintrovae import SoftIntroVAE
register_model("sintrovae", SoftIntroVAE)
from .ali import ALI
register_model("ali", ALI)
from .avb import AdversarialVariationalBayes
register_model("avb", AdversarialVariationalBayes)
from .dagmm import DAGMM
register_model("dagmm", DAGMM)

from .twostagevae import TwoStageVAE
register_model("twostagevae", TwoStageVAE)
from .vamppriorvae import VampPriorVAE
register_model("vamppriorvae", VampPriorVAE)
from .geco import GECO
register_model("geco", GECO)
from .exactelbovae import ExactELBOVAE
register_model("exactelbovae", ExactELBOVAE)
from .iwae import IWAE
register_model("iwae", IWAE)
from .swae import SWAE
register_model("swae", SWAE)
from .tcwae import TCWAE
register_model("tcwae", TCWAE)
from .wvi import WVI
register_model("wvi", WVI)