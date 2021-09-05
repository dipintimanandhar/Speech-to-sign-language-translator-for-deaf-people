
# coding: utf-8

# In[1]:

from __future__ import division 
import numpy as np 
import scipy.io.wavfile as wav
from features import mfcc


# In[2]:

class TestingNetwork:

	layerCount = 0;
	shape  = None;
	weights = [];

	def __init__(self,layerSize,weights):

		self.layerCount = len(layerSize) - 1;
		self.shape = layerSize

		self._layerInput = []
		self._layerOutput = []
		self.weights = weights
	def forwardProc(self,input):

		InCases = input.shape[0]

		self._layerInput = []
		self._layerOutput = []

		for index in range(self.layerCount):
			if index == 0:
				layerInput = self.weights[0].dot(np.vstack([input.T,np.ones([1,InCases])]))
			else:
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,InCases])]))

			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sgm(layerInput))

		return self._layerOutput[-1].T

	def sgm(self,x,Derivative=False):
		if not Derivative:
			return 1/ (1+np.exp(-x))
		else:
			out = self.sgm(x)
			return out*(1-out)


# In[27]:

def testInit():
#Setup Neural Network/
    #f1 = open("C:/Users/Asok/Downloads/majorproject/network/vowel_network_words.npy", 'rt', encoding='latin1')
    #f1 = open("network/vowel_network_words.npy", "rb")
    weights  = np.load('network/vowel_network_words.npy',encoding='latin1')
    testNet = TestingNetwork((260,25,25,5),weights)
    return testNet


        


# In[48]:

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

def extractFeature(soundfile):
    (rate,sig) = wav.read("test_files/test.wav")
    mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.01)
    print("MFCC Feature Length: " + str(len(mfcc_feat)))
    fbank_feat = logfbank(sig,rate)
    s = mfcc_feat[:20]
    st = []
    for elem in s:
        st.extend(elem)
    st /= np.max(np.abs(st),axis=0)
    inputArray = np.array([st])
    return inputArray




# In[49]:

print(inputArray)


# In[5]:

def feedToNetwork(inputArray,testNet):
	#Input MFCC Array to Network
	outputArray = testNet.forwardProc(inputArray)

	#if the maximum value in the output is less than
	#the threshold the system does not recognize the sound
	#the user spoke


	indexMax = outputArray.argmax(axis = 1)[0]
			
	print(outputArray)
	
	#Mapping each index to their corresponding meaning
	outStr = None
	
	if indexMax == 0:
		outStr  = "Detected: Apple"; 
	elif indexMax==1:
		outStr  = "Detected: Banana";
	elif indexMax==2:
		outStr  = "Detected: Kiwi";
	elif indexMax==3:
		outStr  = "Detected: Lime";
	elif indexMax==4:
		outStr  = "Detected: Orange";

	print (outStr)
	return outStr



# In[52]:

if __name__ == "__main__":
    testNet = testInit()
    inputArray = extractFeature("test_files/test.wav")
    feedToNetwork(inputArray,testNet)


# In[50]:





# In[ ]:



