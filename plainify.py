import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import copy
import math


# Global options

# Path to supplementary data
prefix="/PATH/TO/DATA"

# Should GPU be used
usingGPU=False

# Load BERT model

premodel='bert-large-uncased-whole-word-masking'
tokenizer = BertTokenizer.from_pretrained(premodel)
model_dict = torch.load(prefix+"/tersebert/models/nonified/tersebert_pytorch.bin")
model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=premodel, state_dict=model_dict)
vocabulary=tokenizer.get_vocab()
vocabulary2=['']*len(vocabulary)
for s in vocabulary:
	vocabulary2[vocabulary[s]]=s

# Load embeddings

def readWord2Vec(path):
	result={}
	for line in open(path):
		parts=line.strip().split()
		word=parts[0]
		row=np.array([float(x) for x in parts[1:]])
		result[word]=row
	return(result)

def getToken2Vec(embeddingsW):
	result={}
	counter=0
	for word in embeddingsW:
		counter=counter+1
		if counter%100000==0:
			print(str(counter/100000)+"/"+str(len(embeddingsW)/100000))
		tokens=tokenizer.tokenize(word)
		for token in tokens:
			if token in result:
				result[token]=result[token]+embeddingsW[word]
			else:
				result[token]=embeddingsW[word]
	for token in result:
		result[token]=result[token]/np.sqrt(np.sum(result[token]*result[token]))
	vocabulary=tokenizer.get_vocab()
	table=np.zeros((len(vocabulary),len(result['the'])))
	for token in result:
		table[vocabulary[token],]=result[token]
	return(table)

print("Loading Embeddings")
# FastText embeddings, available from https://fasttext.cc/docs/en/english-vectors.html
pathW2V=prefix+"/word2vec/crawl-300d-2M-subword.vec"
embeddingsW=readWord2Vec(pathW2V)
embeddings=getToken2Vec(embeddingsW)
#np.save('./emb.npy',embeddings)
#embeddings=np.load('./emb.npy')
similm=np.matmul(embeddings,np.transpose(embeddings))
print("Loaded Embeddings")

# Load frequencies

def readUnigramFreq(path):
	max=0
	result={}
	for line in open(path):
		parts=line.strip().split()
		word=parts[0]
		frequency=int(parts[1])
		if max==0:
			max=frequency
		result[word]=frequency*1.0/max
	return(result)

def getTokenFreq(unifreq):
	result={}
	counter=0
	for word in unifreq:
		counter=counter+1
		if counter%100000==0:
			print(str(counter/100000)+"/"+str(len(unifreq)/100000))
		tokens=tokenizer.tokenize(word)
		for token in tokens:
			if token in result:
				result[token]=max(result[token],unifreq[word])
			else:
				result[token]=unifreq[word]
	vocabulary=tokenizer.get_vocab()
	table=np.zeros(len(vocabulary))
	for token in result:
		table[vocabulary[token]]=result[token]
	return(table)

print("Loading Unigrams")
pathUni1T=prefix+"/unigrams1T/unigrams-df.tsv"
unifreq=readUnigramFreq(pathUni1T)
tokenfreq=getTokenFreq(unifreq)
#np.save('./freq.npy',tokenfreq)
#tokenfreq=np.load('./freq.npy')
print("Loaded Unigrams")

# Helper functions	

# Tokenise, but keep the word information
def tokeniseUntokenise(sentence):
	tokenised_text = tokenizer.tokenize(sentence)
	is_word=[not str.startswith("##") for str in tokenised_text]
	words=[]
	currentWord=[]
	for i in range(len(is_word)):
		if is_word[i] and currentWord!=[]:
			words.append(currentWord)
			currentWord=[]
		currentWord.append(i)
	words.append(currentWord)
	result={}
	result['tokens']=tokenised_text
	result['words']=words
	return(result)

# Mask the tokens selected through dummies ('_')
def maskDummies(tokenisedOriginal,tokenisedDummies,prefixOriginal,addPAD=0):
	if prefixOriginal:
		maskedTokens=['[CLS]']+tokenisedOriginal['tokens']+['[SEP]']+tokenisedDummies['tokens']+['[SEP]']
		offset=1+len(tokenisedOriginal['tokens'])
	else:
		maskedTokens=['[CLS]']+tokenisedDummies['tokens']+['[SEP]']
		offset=0
	for i in range(len(tokenisedDummies['tokens'])):
		if tokenisedDummies['tokens'][i]=='_':
			maskedTokens[i+1+offset]='[MASK]'
	maskedTokens.extend(['[PAD]']*addPAD)
	return(maskedTokens)

# Prepare input vectors for masked sentences
def prepareVectors(masked,prefixOriginal):
	if not prefixOriginal:
		return (([tokenizer.convert_tokens_to_ids(x) for x in masked],[[0]*len(x) for x in masked]))
	indexed=[]
	segments=[]
	for i in range(len(masked)):
		indexedI=tokenizer.convert_tokens_to_ids(masked[i])
		segmentsI=[]
		currentSegment=0
		for j in range(len(masked[i])):
			segmentsI.append(currentSegment)
			if masked[i][j]=='[SEP]':
				currentSegment=1
		indexed.append(indexedI)
		segments.append(segmentsI)
	return ((indexed,segments))

# Get predictions from BERT
def getPredictions(indexed,segments,prefixOriginal,onGPU):
	tokens_tensor = torch.tensor(indexed)
	segments_tensors = torch.tensor(segments)
	if onGPU:
		tokens_tensor = tokens_tensor.to('cuda')
		segments_tensors = segments_tensors.to('cuda')
		model.to('cuda')
	with torch.no_grad():
		predictions = model(tokens_tensor, segments_tensors)
	if onGPU:
		predictions=(predictions[0].cpu(),)
	if prefixOriginal:
		offset=sum(np.array(segments[0])==0)-1
		predictions=(predictions[0][:,offset:,:],)
	return(predictions)

# Replace a word at a given location with a given token
def replaceWord(tokenised,word,replacement):
	result={'tokens':[],'words':[]}
	offset=0
	for i in range(len(tokenised['words'])):
		if i==word:
			if replacement is None:
				offset=len(tokenised['words'][i])
			else:
				result['tokens'].append(replacement)
				result['words'].append([tokenised['words'][i][0]])
				offset=len(tokenised['words'][i])-1
		else:
			result['tokens'].extend([tokenised['tokens'][x] for x in tokenised['words'][i]])		
			result['words'].append([x-offset for x in tokenised['words'][i]])
	return(result)

# Insert a dummy just before a given position
def insertDummy(tokenised,word):
	result={'tokens':[],'words':[]}
	offset=0
	for i in range(len(tokenised['words'])):
		if i==word:
			result['tokens'].append('_')
			result['words'].append([tokenised['words'][i][0]])
			offset=1
		result['tokens'].extend([tokenised['tokens'][x] for x in tokenised['words'][i]])		
		result['words'].append([x+offset for x in tokenised['words'][i]])
	if word==len(tokenised['words']):
		# Add dummy at the end
		result['tokens'].append('_')
		result['words'].append([len(tokenised['tokens'])])
	return(result)


# Convert tokens to words
def getWords2(tokens):
	if tokens==[]:
		return(tokens)
	words=[]
	for token in tokens:
		if words==[]:
			words.append(token)
		elif not token.startswith("##"):
			words.append(token)
		else:
			words[len(words)-1]=words[len(words)-1]+token.replace("##","")
	return(words)

# Get similarity between two sets of token identifiers by aligning them according to similarity
def getSimilarityAligned(oldTids,newTids):
	subM=similm[oldTids,]
	part1=np.amax(subM,0)
	subM=np.copy(subM)
	if len(newTids)>0:
		for i in range(len(oldTids)):
			best=np.max(subM[i,newTids])
			subM[i,subM[i,]<best]=best
	result=(np.sum(subM,0)+part1+np.sum(part1[newTids]))/(len(oldTids)+len(newTids)+1)
	return(result)

def getSimilarityAveraged(oldTids,newTids):
	oldVector=np.sum(embeddings[oldTids],axis=0)/(len(oldTids))
	oldVector=oldVector/np.sqrt(np.sum(oldVector*oldVector))
	newVectors=np.sum(embeddings[newTids],axis=0)
	mat=embeddings+newVectors
	mat=mat/np.sqrt(np.sum(mat*mat,axis=1))[:,np.newaxis]
	result=np.matmul(mat,oldVector)
	return(result)

# Obtain BERT predictions in a current context
def replacementPredictions(tokenised,currentTokenised,mweStart,prefixOriginal,usingGPU,maxDepth):
	# Prepare prediction task for single mask
	withdummies=insertDummy(currentTokenised,mweStart)
	masked1=maskDummies(tokenised,withdummies,prefixOriginal,1)
	if maxDepth>1:
		# and double mask
		withdummies=insertDummy(withdummies,mweStart)
		masked2=maskDummies(tokenised,withdummies,prefixOriginal,0)
		masked=[masked1,masked2]
	else:
		masked=[masked1]
	indexed,segments=prepareVectors(masked,prefixOriginal)
	predictions=getPredictions(indexed,segments,prefixOriginal,usingGPU)
	return(predictions)

# Obtain BERT predictions in a current context
def multiReplacementPredictions(tokenised,currentTokenised,mweStart,prefixOriginal,usingGPU,maxDepth):
	masked=[]
	for key in currentTokenised:
		# Prepare prediction task for single mask
		withdummies=insertDummy(currentTokenised[key],mweStart)
		masked1=maskDummies(tokenised,withdummies,prefixOriginal,1)
		masked.append(masked1)
		if maxDepth>1:
			# and double mask
			withdummies=insertDummy(withdummies,mweStart)
			masked2=maskDummies(tokenised,withdummies,prefixOriginal,0)
			masked.append(masked2)
	indexed,segments=prepareVectors(masked,prefixOriginal)
	predictions=getPredictions(indexed,segments,prefixOriginal,usingGPU)
	result={}
	i=0
	for key in currentTokenised:
		if maxDepth>1:
			result[key]=(predictions[0][i:(i+2)],)
			i=i+2
		else:
			result[key]=(predictions[0][i:(i+1)],)
			i=i+1
	return(result)

# Compute how good the current fragment is for being a replacement
def getGoodness(current,origTids,scores,alpha=(1,1,1)):
	# Probability of previous tokens times that of candidates
	scoresVec=scores*current['score']
	# Frequency of candidates, unless higher than of previous tokens
	freqVec=np.copy(tokenfreq)
	freqVec[freqVec>current['freqs']]=current['freqs']
	# Similarity taking into account previous tokens
	#similVec=getSimilarityAveraged(origTids,tokenizer.convert_tokens_to_ids(current['tokens']))
	similVec=getSimilarityAligned(origTids,tokenizer.convert_tokens_to_ids(current['tokens']))
	finalGoodness=(scoresVec**alpha[0])*(similVec**alpha[1])*(freqVec**alpha[2])
	return (np.transpose(np.array([finalGoodness,scoresVec,similVec,freqVec])))

# Aggregate results from two sources, i.e. running Plainifier forwards and backwards
def aggregateResults(results):
	allWordsList=[]
	allWordsSet=set()
	allScores=None
	for words,scores in results:
		if allWordsList==[]:
			allWordsList=words.copy()
			allWordsSet.update(words)
			allScores=scores
		else:
			newWordsI=[not(word in allWordsSet) for word in words]
			newWords=[word for word in words if not(word in allWordsSet)]
			allWordsList.extend(newWords)
			allWordsSet.update(newWords)
			allScores=np.concatenate((allScores,scores[newWordsI,]))
	sortedTop=((-allScores[:,0]).argsort())
	return(([allWordsList[i] for i in sortedTop],allScores[sortedTop,]))

# Get replacements for the specified words
def getTokenReplacement(tokenised,mweStart,mweLength,current=None,verbose=False,backwards=False,none_threshold=0.5,maxDepth=3,maxBreadth=16,alpha=(1,1,1)):
	prefixOriginal = True
	# If no depth left, return empty
	if maxDepth==0:
		return (([],[]))
	# Handle root call
	head=False
	if current is None:
		head=True
		current={}
		current['tokenised']=copy.deepcopy(tokenised)
		# Remove the target words
		for i in range(mweLength):
			current['tokenised']=replaceWord(current['tokenised'],mweStart,None)
		current['tokens']=[]
		current['score']=1.0
		current['freqs']=1.0
		current['origTids']=tokenizer.convert_tokens_to_ids(tokenised['tokens'][(tokenised['words'][mweStart][0]):(tokenised['words'][mweStart+mweLength][0])])
		current['guaranteedTids']=current['origTids']
	# Run the predictions, unless precomputed above
	if 'predictions' in current:
		predictions=current['predictions']
	else:
		predictions=replacementPredictions(tokenised,current['tokenised'],mweStart,prefixOriginal,usingGPU,maxDepth)
	# Handle single mask results
	scores=predictions[0][0][current['tokenised']['words'][mweStart][0]+1].numpy()
	scores=np.exp(scores)/sum(np.exp(scores))
	none_score=scores[tokenizer.convert_tokens_to_ids("[unused0]")]
	if verbose:
		print(("\t"*(3-maxDepth))+"NONE: "+str(none_score))
	# If none threshold exceeded, return empty
	if none_score>none_threshold and not head:
		maxDepth=0
	# Compute goodness
	goodness=getGoodness(current,current['origTids'],scores,alpha=alpha)
	if backwards:
		resultTokens=[[vocabulary2[i]]+current['tokens'] for i in range(len(vocabulary2))[999:]]
	else:
		resultTokens=[current['tokens']+[vocabulary2[i]] for i in range(len(vocabulary2))[999:]]
	resultGoodness=goodness[999:,]
	if verbose:
		for i in (-goodness[:,0]).argsort()[:5]:
			print(("\t"*(3-maxDepth))+vocabulary2[i]+": "+str(resultGoodness[i,0]))
	if maxDepth>1 and maxBreadth>0:
		# Get double-word replacements
		if backwards:
			scores=predictions[0][1][current['tokenised']['words'][mweStart][0]+2].numpy()
		else:
			scores=predictions[0][1][current['tokenised']['words'][mweStart][0]+1].numpy()
		scores=np.exp(scores)/sum(np.exp(scores))
		# Compute goodness
		goodness=getGoodness(current,current['origTids'],scores,alpha=alpha)
		# Choose the path
		sortedTop=(-goodness[:,0]).argsort()[0:maxBreadth]
		if len(current['guaranteedTids'])>0:
			if backwards:
				guaranteedTid=current['guaranteedTids'][-1]
				current['guaranteedTids']=current['guaranteedTids'][:-1]
			else:
				guaranteedTid=current['guaranteedTids'][0]
				current['guaranteedTids']=current['guaranteedTids'][1:]
			if not (guaranteedTid in sortedTop):
				scores[guaranteedTid]=scores[sortedTop[-1]]
				sortedTop[-1]=guaranteedTid
		# Prepare data for recursive execution
		currentNews={}
		for i in sortedTop:
			bestCandidate=vocabulary2[i]
			if bestCandidate=="[unused0]":
				continue
			newsentence=insertDummy(current['tokenised'],mweStart)
			newsentence=replaceWord(newsentence,mweStart,bestCandidate)
			currentNew={}
			currentNew['tokenised']=newsentence
			currentNew['origTids']=current['origTids']
			currentNew['guaranteedTids']=current['guaranteedTids']
			if backwards:
				currentNew['tokens']=[bestCandidate]+current['tokens']
			else:
				currentNew['tokens']=current['tokens']+[bestCandidate]
			currentNew['score']=current['score']*scores[i]
			currentNew['freqs']=min(current['freqs'],tokenfreq[i])
			currentNews[i]=currentNew
		# Pre-compute predictions
		newsentences={i: currentNews[i]['tokenised'] for i in sortedTop}
		if backwards:
			newmweStart=mweStart
		else:
			newmweStart=mweStart+1
		preds=multiReplacementPredictions(tokenised,newsentences,newmweStart,prefixOriginal,usingGPU,maxDepth-1)
		counter=0
		# Run recursively		
		for i in sortedTop:
			bestCandidate=vocabulary2[i]
			if bestCandidate=="[unused0]":
				continue
			if verbose:
				print(("\t"*(3-maxDepth))+str(counter)+"/"+str(maxBreadth)+" "+bestCandidate)
			counter=counter+1
			currentNews[i]['predictions']=preds[i]
			nextTokens,nextGoodness=getTokenReplacement(tokenised, newmweStart,0, currentNews[i],verbose,backwards,none_threshold,maxDepth-1,round(maxBreadth/2),alpha)
			if len(nextTokens)==0:
				continue
			resultTokens=resultTokens+nextTokens
			resultGoodness=np.concatenate((resultGoodness,nextGoodness))
	# If not head call, just return candidates
	if not head:
		return((resultTokens,resultGoodness))
	# Otherwise, rank the candidates
	sortedTop=(-(resultGoodness[:,0])).argsort()
	if verbose:
		for i in range(50):
			print(str(resultGoodness[sortedTop[i]])+" : "+str(resultTokens[sortedTop[i]]))
	resultGoodness=resultGoodness[sortedTop,:]
	resultTokens=[" ".join(getWords2(resultTokens[i])) for i in sortedTop]
	result=(resultTokens,resultGoodness)
	return(result)

sentence=tokeniseUntokenise("The cat perched on the mat.")
result1=getTokenReplacement(sentence,2,1,verbose=True,backwards=False,maxDepth=3,maxBreadth=16,alpha=(1/9,6/9,2/9))
result2=getTokenReplacement(sentence,2,1,verbose=True,backwards=True,maxDepth=3,maxBreadth=16,alpha=(1/9,6/9,2/9))
words,scores=aggregateResults((result1,result2))
list(zip(words[:10],scores[:10]))
# [('was', array([0.27730091, 0.02695862, 0.26732088, 0.9927352 ])), ('lying', array([0.27037246, 0.0314074 , 0.29708439, 0.59796472])), ('sitting', array([0.26936967, 0.0134001 , 0.34319147, 0.58399022])), ('landed', array([0.26843866, 0.00603219, 0.45939506, 0.35727813])), ('out', array([0.26773613, 0.01323677, 0.28579689, 0.98996078])), ('down', array([0.26472962, 0.0107758 , 0.29466948, 0.95143947])), ('lay', array([0.26181015, 0.02171396, 0.28032156, 0.7406279 ])), ('lay down', array([0.2489916 , 0.01130033, 0.28988684, 0.7406279 ])), ('stood', array([2.48593001e-01, 5.98142098e-04, 4.96567600e-01, 6.35861778e-01])), ('perched', array([2.45833581e-01, 1.62489101e-04, 1.00000000e+00, 1.42060519e-01]))]
















