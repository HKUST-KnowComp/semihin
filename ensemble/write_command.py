
DIFF = ['rec.autos', 'comp.os.ms-windows.misc', 'sci.space']
SIM = ['comp.graphics', 'comp.sys.mac.hardware', 'comp.os.ms-windows.misc']
GSIM = ['GWEA','GDIS','GENV']
GDIF = ['GENT','GODD','GDEF']

scope_name = 'DIFF'
scope = DIFF

command_file = open('/home/hejiang/code/get-another-label/bin/' + scope_name + '.sh','w')
for lb in range(1,11):
    for r in range(50):
        # run get-another-label batch
        command = r'/home/hejiang/code/get-another-label/bin/get-another-label.sh ' + \
                  '--cost /home/hejiang/results/gal/' + scope_name + '_costs.txt ' + \
                  '--gold /home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + \
                  '_' + str(r).zfill(3) + '_gold.txt ' + \
                  '--eval /home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + \
                  '_' + str(r).zfill(3) + '_eval.txt ' + \
                  '--categories /home/hejiang/results/gal/' + scope_name + '_categories.txt ' + \
                  '--input /home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + \
                  '_' + str(r).zfill(3) + '_label.txt ' + \
                  '> /home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + '_' + \
                  str(r).zfill(3) + '_result.txt'

        command_file.write(command + '\n')

scope_name = 'SIM'
scope = SIM

command_file = open('/home/hejiang/code/get-another-label/bin/' + scope_name + '.sh','w')
for lb in range(1,11):
    for r in range(50):
        # run get-another-label batch
        command = r'/home/hejiang/code/get-another-label/bin/get-another-label.sh ' + \
                  '--cost /home/hejiang/results/gal/' + scope_name + '_costs.txt ' + \
                  '--gold /home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + \
                  '_' + str(r).zfill(3) + '_gold.txt ' + \
                  '--eval /home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + \
                  '_' + str(r).zfill(3) + '_eval.txt ' + \
                  '--categories /home/hejiang/results/gal/' + scope_name + '_categories.txt ' + \
                  '--input /home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + \
                  '_' + str(r).zfill(3) + '_label.txt ' + \
                  '> /home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + '_' + \
                  str(r).zfill(3) + '_result.txt'

        command_file.write(command + '\n')