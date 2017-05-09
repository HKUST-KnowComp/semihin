import numpy as np
DIFF = ['rec.autos', 'comp.os.ms-windows.misc', 'sci.space']
SIM = ['comp.graphics', 'comp.sys.mac.hardware', 'comp.os.ms-windows.misc']
GSIM = ['GWEA','GDIS','GENV']
GDIF = ['GENT','GODD','GDEF']

scope_name = 'DIFF'
scope = DIFF

print 'DIFF'
for lb in range(1,11):
    results = []
    for r in range(50):
        # read get-another file
        result_file = open('data/' + scope_name + '_lb' + str(lb).zfill(3) + '_' +
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

print 'SIM'
scope_name = 'SIM'
for lb in range(1,11):
    results = []
    for r in range(50):
        # read get-another file
        result_file = open('/home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + '_' +
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
