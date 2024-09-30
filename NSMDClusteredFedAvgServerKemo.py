from collections import OrderedDict
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_distances
# from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# from scipy.spatial.distance import mahalanobis
from sklearn.metrics.cluster import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr as fl
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import aggregate, aggregate_inplace, weighted_loss_avg
from scipy.spatial.distance import mahalanobis
import torch
import torch.nn as nn
from torch import save
from torch.utils.data import DataLoader
from Old_train import valid, make_model_folder
import warnings
from utils import *
from Network import *
import os
import numpy as np
import random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Federatedparser()
# eval_data = K_EMODataset()
# eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False, collate_fn= lambda x:x)
lossf = nn.BCEWithLogitsLoss()
net = K_emo_GRU(3, 4, 1)
keys = net.state_dict().keys()
if args.pretrained is not None:
    net.load_state_dict(torch.load(args.pretrained))
net.to(DEVICE)
# pca = PCA(2)
kmeans = KMeans(n_clusters=2)
kmeans2 = KMeans(n_clusters=2, n_init="auto")
mode = 1 if args.mode == "max" else 0
CosVsMaha={"cosine":[], "mahalanobis":[], "cosine_sil":[], "mahalanobis_sil":[]}
scaler = StandardScaler()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

def set_parameters(net, parameters):
    try:
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(state_dict)
        print(e)


def fl_evaluate(server_round:int, parameters: fl.common.NDArrays, config:Dict[str, fl.common.Scalar],)-> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    set_parameters(net, parameters)
    history=valid(net, eval_loader, None, lossf, DEVICE)
    save(net.state_dict(), f"./Models/{args.version}/net.pt")
    print(f"Server-side evaluation loss {history['loss']} / accuracy {history['acc']} / precision {history['precision']} / f1score {history['f1score']}")
    return history['loss'], {key:value for key, value in history.items() if key != "loss" }

def fl_save(server_round:int, parameters: fl.common.NDArrays, config:Dict[str, fl.common.Scalar],)-> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    set_parameters(net, parameters)
    save(net.state_dict(), f"./Models/{args.version}/net.pt")
    print("model is saved")
    return 0, {}

def cosine_distance_cal(X,Y):
    cosine = sum([cosine_distances(x.reshape((1,-1)),y.reshape((1,-1))) for x,y in zip(X,Y)])
    return cosine

def mahalanobis_distance_cal(X):
    mahala = calculateMahalanobis(X[0], X[1])
    if mahala != np.NAN:
        return mahala 
    else:
        print("마하라노비스 Nan값입니다.")
        return 0
    
def calculateMahalanobis(Y=None, data=None):
    covar=[[[np.cov(d,y, ddof=False, rowvar=True)[0,1]]] for d,y in zip(data,Y)]
    try:
        return [mahalanobis(dy.reshape([-1,]),dx.reshape([-1,]), np.linalg.inv(c)) for y,d,c in zip(Y, data, covar) for dx,dy in zip(d,y)]
    except Exception as e :
        if str(e) == "Singular matrix":
            print(e)
            return [mahalanobis(dy.reshape([-1,]),dx.reshape([-1,]), c) for y,d,c in zip(Y, data, covar) for dx,dy in zip(d,y)]
        else:
            print(e)
            return sum([sum(y-d) for y,d in zip(Y, data)])

def parameter_to_Ndarrays(param):
    return [v.flatten() for v in param]

def parameter_Dnumber_samples(param, num_examples, sum_of_exmaples):
    
    return [np.array(list(map(lambda x: x*num_examples/sum_of_exmaples, p))) for p in param]

class ClusteredFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *, fraction_fit: float = 1, fraction_evaluate: float = 1, min_fit_clients: int = 2, min_evaluate_clients: int = 2, min_available_clients: int = 2, evaluate_fn, on_fit_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, on_evaluate_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, accept_failures: bool = True, initial_parameters: Parameters | None = None, fit_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, evaluate_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, inplace: bool = True) -> None:
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy | FitRes]], failures: List[Tuple[ClientProxy | FitRes] | BaseException]) -> Tuple[Parameters | None | Dict[str, bool | bytes | float | int | str]]:
        clusters={}
        clusters2={}
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")
            
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            
            """Extract Parameter for distances """
            only_params = [parameters_to_ndarrays(fit_res.parameters)
                for _, fit_res in results]
            sum_of_sample = sum([fit_res.num_examples
                for _, fit_res in results])
            params_of_numfeature = [parameter_Dnumber_samples(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, sum_of_sample)
                for _, fit_res in results]
            
            
            aggregated_ndarrays = aggregate(weights_results)
            
            '''cosine Clustering Part'''
            for indx, client_params in  enumerate([parameter_to_Ndarrays(params) for params in only_params]):
                clusters[f"client{indx+1}"] = cosine_distance_cal(parameter_to_Ndarrays(client_params), parameter_to_Ndarrays(aggregated_ndarrays))
            '''mahalanobis Clustering Part'''
            for indx, client_params in  enumerate([parameter_to_Ndarrays(params) for params in params_of_numfeature]):
                clusters2[f"client{indx+1}"] = mahalanobis_distance_cal((parameter_to_Ndarrays(client_params), parameter_to_Ndarrays(aggregated_ndarrays)))
            
            
            cluster_indexs = kmeans.fit_predict(np.array(list(clusters.values())).reshape((-1,1)))
            cluster_indexs2 = kmeans2.fit_predict(np.array(list(clusters2.values())).reshape((indx+1, -1)))
            
            print(cluster_indexs)
            print(cluster_indexs2)
            CosVsMaha["cosine"].append(kmeans.inertia_)
            CosVsMaha["mahalanobis"].append(kmeans2.inertia_)
            try:
                CosVsMaha["cosine_sil"].append(silhouette_score(np.array(list(clusters.values())).reshape((indx+1,-1)), cluster_indexs.reshape((indx+1,1))))
            except Exception as e:
                print(e)
                CosVsMaha["cosine_sil"].append(1)
            try:
                CosVsMaha["mahalanobis_sil"].append(silhouette_score(np.array(list(clusters2.values())).reshape((indx+1,-1)), cluster_indexs2.reshape((indx+1,1))))
            except Exception as e:
                print(e)
                CosVsMaha["mahalanobis_sil"].append(1)
            n1 = np.count_nonzero(cluster_indexs2)
            n0 = indx+1-n1
            if n1==0 or n0 ==0:
                parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
                return parameters_aggregated, metrics_aggregated
            
            if n1> n0:
                indexs = np.arange(len(cluster_indexs2))[cluster_indexs2==mode]
            else:
                indexs = np.arange(len(cluster_indexs2))[cluster_indexs2== 0 if mode ==1 else 1]
                
            weights_results=[weights_results[i] for i in indexs]
            aggregated_ndarrays = aggregate(weights_results)
            
            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)


        return parameters_aggregated, metrics_aggregated
        
        
if __name__=="__main__":
    warnings.filterwarnings("ignore")
    
    make_model_folder(f"./Models/{args.version}")
    history=fl.server.start_server(server_address='[::]:8084',strategy=ClusteredFedAvg(evaluate_fn=fl_save, inplace=False, min_fit_clients=5, min_available_clients=5, min_evaluate_clients=5), 
                           config=fl.server.ServerConfig(num_rounds=args.round))
    # plt=pd.DataFrame(history.losses_distributed, index=None)[1]
    # plt.plot().figure.savefig(f"./Plot/{args.version}_loss.png")
    # plt.to_csv(f"./Csv/{args.version}_loss.csv")
    performance_cluseter = pd.DataFrame(CosVsMaha, index=None)[1:]
    # performance_cluseter.plot().figure.savefig(f"./Plot/{args.version}_cluster.png")
    performance_cluseter.to_csv(f"./Csv/{args.version}_cluster.csv")