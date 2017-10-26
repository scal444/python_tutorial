import sys


infile  = sys.argv[1] # 0 is the script name
outfile = sys.argv[2]


with open(infile,'r') as fin, open(outfile,'w') as fout:
    for line in fin:
        num = int(line)
        fout.write(str(num ** 2) + '\n')  # square and write out

exit()
