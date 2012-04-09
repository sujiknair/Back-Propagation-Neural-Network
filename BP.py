import math
import sys

class BP:

   # constructor of the class
   def __init__(self):
      print '\n'
      print """ This is a Feed forward neural network (three layer) implementation using Back propagation learning algorithm."""
      print '\n'

   # used for reading numbers from a string
   def read_num_line(self,line):
	l=0;
	h=len(line)
	b=list()
	for i in range(0,len(line)):
           if( (line[i]=='\t')|(line[i]==' ')|(line[i]=='\n')):
              b.append(float(line[l:i]))
	      l=i+1
	return b

   # used for initialization of the data fields
   def initialize(self,s):
        bin_list=list()
	bi=file(sys.argv[2])
	while True:
	   line=bi.readline()
	   if len(line)==0:
		break
	   bin=self.read_num_line(line)
	   bin_list.append(bin)
	bi.close()
	f=file(s)
	x_list=list()
	y_id=list()
	while True:
	   line=f.readline()
	   if len(line)==0:
	      break
	   x=self.read_num_line(line)
	   in_dim=len(x)
           x_list.append(x[1:])
	   for i in range(0, len(bin_list)):
	      if x[0]>= bin_list[i][0]:
	         if x[0]<=bin_list[i][1]:
		    l=list()
		    for s in range(0,len(bin_list)):
		       if s==i:
		          l.append(1)
		       else:
			  l.append(0)
		    y_id.append([x[0],l])
		
	f.close()
	return y_id,x_list,bin_list

   # computing the hidden layer output and the output layer output.
   def compute(self,w_hidden,w_out,x_list,bin_list):
	x_hidden=list()
	y_out=list()
	z=0
	for j in range(0,len(w_hidden)):
	   for k in range(0,len(w_hidden[0])):
		z=z+x_list[k]*w_hidden[j][k]
	   x_hidden.append(1/(1+math.exp(-(z))))	
	check=0
	for i in range(0,len(w_out)):
	   for j in range(0,len(w_hidden)):
		check=check+x_hidden[j]*w_out[i][j]
	   y_out.append(1/(1+math.exp(-(check))))
	  # y_out.append(1/(1+math.exp(-(check)))+math.tanh(-check))
	   #y_out.append(check-10)
	return x_hidden,y_out	

   # The Back propagation learning algorithm
   def Back_Prop(self,w_hidden,w_out,x_hidden,y_out,x_list,eta,y_id):
	
	for i in range(0,len(w_out)):
	       for j in range(0,len(w_out[0])):
		  w_out[i][j]=w_out[i][j]-eta*(y_out[i]-y_id)*y_out[i]*(1-y_out[i])*x_hidden[j]
	
	for j in range(0,len(w_hidden)):
	   for k in range(0,len(w_hidden[0])):
		m=0
		for i in range(0,len(w_out)):
			m=m+(y_out[i]-y_id)*y_out[i]*(1-y_out[i])*w_out[i][j]
		w_hidden[j][k]=w_hidden[j][k]-eta*m*x_hidden[j]*(1-x_hidden[j])*x_list[k]

	return w_hidden,w_out
   
   def train(self):
        y_id_train,x_list_train,bin_list=self.initialize(sys.argv[1])
	bin_num=len(bin_list)
	print "Enter the error maximum"
	err_max=int(raw_input())
	#err_max=len(x_list_train)*er/100
	print "Enter the number of hidden units (Number of nodes in the hidden layer)"
	print "Number of output nodes and input nodes are"+' '+str(bin_num)+' '+str(len(x_list_train[0]))
	num_hidden=int(raw_input())
	num_input=len(x_list_train[0])
	print "Enter the learning constant"
	eta=float(raw_input())
	w_hidden=list()
	w_out=list()
	for j in range(0,num_hidden):
	   w=list()
	   for k in range(0,num_input):
		w.append(float(0.0))
	   w_hidden.append(w)
	for i in range(0,bin_num):
	   w=list()
	   for j in range(0,num_hidden):
		w.append(-(float(0.4)))
	   w_out.append(w)

	ep=1
	tr=file("train_out.txt","w")
	while True:
	   err=0
	   for i in range(0,len(x_list_train)):
	      x_hidden,y_out=self.compute(w_hidden,w_out,x_list_train[i],bin_list)
	      err_inter=0
	      for m in range(0,bin_num):
	      	 err_inter=(y_out[m]-y_id_train[i][0])**2
	      err=err+(1/2)*math.sqrt(err_inter)
	      w_hidden,w_out=self.Back_Prop(w_hidden,w_out,x_hidden,y_out,x_list_train[i],eta,y_id_train[i][0])
	      tr.write("W values at epoch "+str(ep)+" and iteration "+str(i)+" is "+'\n\n')
	      tr.write("W_hidden\n")
	      tr.write(str(w_hidden)+'\n\n')
	      tr.write("W_out\n")
	      tr.write(str(w_out)+'\n\n')
	   ep=ep+1 
	  # if err<=err_max:
	#	break
	   if ep==10:
		break
		
	#print w_hidden
	#print w_out
	tr.close() 
	return w_hidden,w_out
   def test(self,w_hidden,w_out):
	te=file("test_out.txt","w")
	y_id_test,x_list_test,bin_list=self.initialize(sys.argv[3])
	for i in range(0,len(x_list_test)):
	   x_hidden,y_out=self.compute(w_hidden,w_out,x_list_test[i],bin_list)
	   te.write("Output for the test "+str(i)+" is "+str(y_out)+'\n')
	   te.write("Desired value is "+str(y_id_test[i][0])+' '+str(y_id_test[i][1])+'\n')
	te.close()
if len(sys.argv)<4:
	print "Execution format is <executable> <training file> <classfication file> <test file>" 
	sys.exit()

if __name__=='__main__':

	b=BP()
	w_hidden,w_out=b.train()
	print "Output of training is written in to train_out.txt"
	b.test(w_hidden,w_out)
	print "Output of testing is written in to test_out.txt"

				
