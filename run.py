from main import run_semihin
from main import run_lp
from main import run_lp_meta
from main import run_knowsim
from main import run_svm
from main import run_nb
from main import run_meta_graph_ensemble_svm
from main import run_meta_graph_ensemble_gal
from main import run_meta_graph_ensemble_cotrain
from main import dump_hin
from main import generate_train_test_split
from main import run_laplacian_feature_search
from main import run_generate_meta_graph
from main import print_result
from main import dump_result
from main import run_generate_laplacian_score

def run_all_experiments():
    run_semihin()
    run_lp()
    run_lp_meta()
    run_knowsim()
    run_svm()
    run_nb()
    run_meta_graph_ensemble_svm()
    run_meta_graph_ensemble_gal()
    run_meta_graph_ensemble_cotrain()


def run_all():
    dump_hin()
    generate_train_test_split()
    run_laplacian_feature_search()
    run_generate_meta_graph()
    run_all_experiments()
    print_result()
    dump_result()

#run_generate_laplacian_score()
#run_generate_meta_graph()
#run_meta_graph_ensemble_svm()
#run_meta_graph_ensemble_gal()
run_meta_graph_ensemble_cotrain()
print_result()
