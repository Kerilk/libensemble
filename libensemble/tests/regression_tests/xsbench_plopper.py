import os, sys, subprocess, random

class Plopper:
    def __init__(self,sourcefile,outputdir):
        self.sourcefile = sourcefile
        self.outputdir = outputdir+"/tmp_files"

        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

    def createDict(self, x, params):
        dictVal = {}
        for p, v in zip(params, x):
            dictVal[p] = v
        return(dictVal)

    def plotValues(self, dictVal, inputfile, outputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()

        with open(outputfile, "w") as f2:
            for line in buf:
                modify_line = line
                for key, value in dictVal.items():
                    if key in modify_line:
                        if value != 'None': #For empty string options
                            modify_line = modify_line.replace('#'+key, str(value))

                if modify_line != line:
                    f2.write(modify_line)
                else:
                    f2.write(line)

    def findRuntime(self, x, params, worker):
        interimfile = ""
        exetime = 1
        counter = random.randint(1, 10001) # To reduce collision increasing the sampling intervals
        interimfile = self.outputdir+"/"+str(counter)+".c"

        # Generate intermediate file
        dictVal = self.createDict(x, params)
        self.plotValues(dictVal, self.sourcefile, interimfile)

        #compile and find the execution time
        tmpbinary = interimfile[:-2] + '_w' + str(worker)
        kernel_idx = self.sourcefile.rfind('/')
        kernel_dir = self.sourcefile[:kernel_idx]

        gcc_cmd = "mpicc -std=gnu99 -Wall -flto  -fopenmp -DOPENMP -DMPI -O3 "  + \
        " -o " + tmpbinary + " " + interimfile +" " + kernel_dir + "/Materials.c " \
        + kernel_dir + "/XSutils.c " + " -I" + kernel_dir + \
        " -lm" + " -L${CONDA_PREFIX}/lib"
        run_cmd = kernel_dir + "/exe.pl " +  tmpbinary

        compilation_status = subprocess.run(gcc_cmd, shell=True, stderr=subprocess.PIPE)

        #Find the execution time only when the compilation return code is zero, else return infinity
        if compilation_status.returncode == 0 :
            execution_status = subprocess.run(run_cmd, shell=True, stdout=subprocess.PIPE)
            exetime = float(execution_status.stdout.decode('utf-8'))
            if exetime == 0:
                exetime = 1
        else:
            print(compilation_status.stderr)
            print("compile failed")
        return exetime #return execution time as cost
