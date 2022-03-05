from PSO_model.tools import set_run_mode
import datetime
from PSO_model.PSO import PSO
from modelRuning import runing
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=859)


def schaffer(x):
    '''
    PSO objective function
    '''
    mu_t,delta_t,kappa_t,lamb_t = x[0]
    pop_index = x[1]
    options = x[2]
    mu_t = round(mu_t, 1)
    delta_t = round(delta_t, 1)
    kappa_t = round(kappa_t, 1)
    lamb_t = round(lamb_t, 1)
    options["mu"] = mu_t
    options["delta"] = delta_t
    options["kappa"] = kappa_t
    options["lamb"] = lamb_t
    FMI,ARI,NMI= runing(options, options["training_set"], options["testing_set"], mode=options["mode"])
    print("pop_index---"+str(pop_index)+"\tmu---"+str(mu_t)+"\tdelta_t---"+str(delta_t)+"\tkappa_t---"+str(kappa_t)+"\tlamb_t---"+str(lamb_t)+"\t||\tFMI---"+str(FMI)+"\tARI---"+str(ARI)+"\tNMI---"+str(NMI))
    cost = 1.0 / (FMI+1)
    return cost


def PSO_start(options,m = "parallel", processNum = 1,logger=None):

    if processNum > 1:
        set_run_mode(schaffer, m, processNum)
    print("-"*80)
    print("Starting the PSO optimal model for {} dataset.".format(options["dataset"]))
    muLimit = options["mu"]
    deltaLimit = options["delta"]
    kappaLimit = options["kappa"]
    lambLimit = options["lamb"]

    lb =[muLimit[0],deltaLimit[0],kappaLimit[0],lambLimit[0]]
    ub = [muLimit[1], deltaLimit[1], kappaLimit[1], lambLimit[1]]
    pso = PSO(func=schaffer, n_dim=4, pop=8, max_iter=200, lb=lb, ub=ub,options = options)
    start_time = datetime.datetime.now()
    best_x, best_FMI = pso.run(logger=logger)
    end_time = datetime.datetime.now()

    logger.info("PSO total time: "+str((end_time - start_time).total_seconds()))

    logger.info('best_x:' + str(list(best_x)) + '\t'+'best_y:'+str(list(best_FMI)))



    pass