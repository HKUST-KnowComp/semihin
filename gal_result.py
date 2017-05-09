import numpy as np

def read_gal_result(scope_name):
    results = []
    for r in range(50):
        # read get-another file
        result_file = open('data/local/gal/' + scope_name + '/lb' + str(5).zfill(3) + '_' +
                         str(r).zfill(3) + '_result.txt')
        loss = []
        for line in result_file.readlines():
            sp = line.split()
            if line.startswith('[DataCost_Eval_DS_ML]'):
                loss.append(float(sp[9]))
            elif line.startswith('[DataCost_Eval_MV_ML]'):
                loss.append(float(sp[8]))
            elif line.startswith('[DataCost_Eval_DS_Min]'):
                loss.append(float(sp[8]))
            elif line.startswith('[DataCost_Eval_MV_Min]'):
                loss.append(float(sp[8]))
            elif line.startswith('[DataCost_Eval_DS_Soft]'):
                loss.append(float(sp[8]))
            elif line.startswith('[DataCost_Eval_MV_Soft]'):
                loss.append(float(sp[8]))
        results.append(1 - np.min(np.array(loss)))
    print str(np.mean(results)) + '\t' + str(np.std(results))

scopes = ['SIM', 'DIFF', 'GSIM', 'GDIF']
for s in scopes:
    print s + ' gal ensemble'
    read_gal_result(s)