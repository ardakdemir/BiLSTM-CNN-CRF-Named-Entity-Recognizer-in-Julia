
## BiLSTM-CNN-CRF-Named-Entity-Recognizer-in-Julia
##main file 


using Knet
using JLD
#include(Knet.dir("test/gpu.jl"))
#import vocab
#import knetreader

##nocrf updates
## mypred icindeki getbilstm, model tanimindaki weightler 
## model tanimi ve transloss



###VOCAB
##vocab.jl

function tag2i(taglist)
    t2i= Dict()
    #t2i["P"] = 1
    c =1
    for tag in taglist
        t2i[tag] = c
        c= c + 1
    end
    t2i
end
function getsenttaginds(tags,t2i,bucketsize)
    arr1 = Vector{Int64}()
    pad = t2i["O"]
    for tag in tags
        push!(arr1,t2i[tag])
    end
    if length(arr1)>bucketsize
        return arr1[1:bucketsize]
    else
        for i = length(arr1)+1:bucketsize
            push!(arr1,pad)
        end
    end
    return arr1
end
function labeltoonehot(taglist)
    tagtoonehot=Dict()
    length1 = length(taglist)
    c = 1
    arr1 = zeros(10,1)
    for tag in taglist
        arr2 = zeros(10,1)
        arr2[c] = 1
        tagtoonehot[tag] = arr2
        c = c+1
    end
    return tagtoonehot
end
function getonehots(tags,onehotdict)
    onehots = []
    for tag in tags
        push!(onehots,onehotdict[tag])
    end
    onehots
end

function vocab(docs,embeds)
    vocab1=["OOV"]
    f = open(embeds)
    lines = readlines(f)
    pretrains=Dict()
    w2i = Dict{String,Int}("OOV"=>1)
    for line in lines
        linesp = split(line)
        push!(vocab1,linesp[1])
        w2i[linesp[1]] = length(vocab1)
    end
    for doc in docs
        for sent in doc
            for word in sent
                w = lowercase(String(word))
                if get(w2i,w,-1)==-1
                    push!(vocab1,w)
                    w2i[w] = length(vocab1)
                end
            end
        end
    end
    return (vocab1,w2i)
end
function chardict(wordlist)
    i2c = ['0','9']## 0 => padding , 9 => unknown char
    c2i = Dict{Char,Int}('0'=>1 , '9'=>2)
    for word in wordlist
        for ch in word
            cs = ch
            if !haskey(c2i,cs)
                push!(i2c,cs)
                c2i[cs] = length(c2i)
            end
        end
    end
    return (i2c,c2i)
end



## KNET MODEL BiLSTM CNN CRF for NER 
### author arda akdemir 

##generate an array containing documents which contain sentences in all documents
function read_sents(filename)
    docs = []
    sents = []
    sent= []
    f = open(filename)
    lines = readlines(f);
    for line in lines
        if occursin("DOCSTART",line) 
            if length(sents)> 0
                push!(docs,sents)
                sents=[]
                sent=[]
            end
        elseif length(line)==0 ## new sentence
            if length(sent)>0
                push!(sents,sent)
                sent =[]
            end
        else
            push!(sent,split(line)[1])
        end
    end
    docs
end


##read 
function read_with_labels(filename)
    docs = []
    sents = []
    sent = []
    tags = []
    taglist = String[]
    f = open(filename)
    lines = readlines(f);
    for line in lines
        if occursin("DOCSTART",line) 
            if length(sents)> 0
                push!(docs,(sents))
                sents=[]
                sent=[]
            end
        elseif length(line)==0 ## new sentence
            if length(sent)>0
                push!(sents,(sent,tags))
                sent =[]
                tags = []
            end
        else
            push!(sent,split(line)[1])
            push!(tags,split(line)[end])
            if !(split(line)[end] in taglist)
                push!(taglist,split(line)[end])
            end
        end
    end
    push!(docs,sents)
    docs,taglist
end

#docs,taglist = read_with_labels("knetfolder/train.txt");
#taglist

println("I am working")

## PARAMETERS OF THE MODEL
trainingfile = "train.txt"
CEMBDIM = 25
WEMBDIM = 100
CAPEMBDIM = 10
CAPTYPES = 3 #all lower, Title, FULLCAPS,miXed, numeric
WEMBRANGE = sqrt(3/WEMBDIM)
CEMBRANGE = sqrt(3/CEMBDIM)
maxWordCharSize = 20
CNNOutputSize = 30
DROPOUT = 0.5
FILTERSIZE = 3
FILTERNUM = 30

maxWordLength= 20
EMBEDSIZE2 = 140
NUMHIDDEN2 = 200
#OUTPUTSIZE2 = length(taglist)


##save model using jld
function savemodel(jldname,modelname,modelvariable)
	save(jldname,modelname,modelvariable)
end

## initialize weights
function initembs(CEMBDIM,WEMBDIM,CAPEMBDIM;CHARTYPES= length(i2c), WORDTYPES = length(i2w))   
    charembs =CEMBRANGE.*rand(Float32,CEMBDIM,CHARTYPES)
    charMatrix = convert(Array{Float32},charembs.-charembs/2)
    capembs = CEMBRANGE.*rand(Float32,CAPEMBDIM,CAPTYPES)
    capMatrix = convert(Array{Float32},capembs.-capembs/2)
    charMatrix[:,1] = [0 for i=1:CEMBDIM]
    #println("charmat")
    #println(charMatrix[:,1])
    #flush(stdout)
    wordembs = rand(Float32,WEMBDIM,WORDTYPES)
    wordMatrix = convert(Array{Float32},wordembs.-wordembs/2)
    return charMatrix, wordMatrix,capMatrix
end

function initcnnweights(filtersize,filternum,outputsize)
    filtrange = sqrt(6/(2*filtersize))
    filter1 = convert(Array{Float32},filtrange.*xavier(Float32,filtersize,filtersize,1,filternum))
    dim2 = filternum*trunc(Int,(CEMBDIM-filtersize+1)/2) * trunc(Int,(maxWordCharSize-filtersize+1.0)/2)
    fulrange = sqrt(6/(outputsize+dim2))
    fullyconnected = convert(Array{Float32},fulrange.*xavier(Float32,outputsize,dim2))
    fullbias= zeros(Float32,outputsize,1)
    return filter1,fullyconnected,fullbias
end

function getcharinds(word)
    cinds=[]
    for c in word
        push!(cinds,get(c2i,c,2))##unknown
    end
    cinds
end

function getcharmat(charembs1,word)
    charmat = charembs1[:,getcharinds(word)]
    pad = charembs1[:,1]
    #println(length(pad))
    if length(word)>=20
        return charmat[:,1:20]
    end
    
    for i= 1: (21-length(word))/2
        charmat = hcat(pad,charmat)
        charmat = hcat(charmat,pad)
    end
    #println("word")
    #println(word)
    #println(value(charmat))
    #flush(stdout)
    return charmat[:,1:20]
end


function linearizecharvecs(charvecs)
    linear = Vector{Float32}()
    for charvec in charvecs
        linear = vcat(linear,charvec)
    end
    return linear
end
function paddedwordtocharembs(charembs1,word)
    charmat = getcharmat(charembs1,word)
    xfinal = reshape(linearizecharvecs(charmat),(CEMBDIM,maxWordCharSize,1,1))
    return xfinal
end


function cnnpredict(cnnweights,charembs1,word)
    xfinal = dropout(paddedwordtocharembs(charembs1,word),DROPOUT)
    convout = conv4(cnnweights[1],xfinal)
    #pooled1 = pool(convout)
    pooled2 = mat(pool(convout,window=maximum(size(convout))))
    #cnnoutput = cnnweights[2] * mat(pooled1).+cnnweights[3]
    #return cnnoutput
    #println(summary(pooled2))
    #flush(stdout)
    return pooled2
end

function getcapindex(word)##all lower,Title, ALLCAPS
    if word == lowercase(word)
        return 1
    end
    if lowercase(word[1])!=word[1]
        for i=2:length(word)
            if word[i] == lowercase(word[i])
                return 2
            end
        end
        return 3
    end
    return 1
end
               

function getwordvector(cnnweights1,charembs1,wordembs1,capembs1,word)
    #println("ne donuyor")
    #println(summary(capembs1[:,getcapindex(word)]))
    #flush(stdout)
    vcat(wordembs1[:,get(w2i,lowercase(word),1)],cnnpredict(cnnweights1,charembs1,word),capembs1[:,getcapindex(word)])
end


function getsentvec(cnnweights1,charembs1,wordembs1,capembs1,sent,bucketsize)
    sentvec = getwordvector(cnnweights1,charembs1,wordembs1,capembs1,sent[1])
    for word in sent[2:end]
        vec = getwordvector(cnnweights1,charembs1,wordembs1,capembs1,word)
        #println(getwordvector(word))
        sentvec = hcat(sentvec,vec)
    end
    if size(sentvec)[2]>bucketsize
        return sentvec[:,1:bucketsize]
    else
        for i = size(sentvec)[2]+1:bucketsize
            sentvec = hcat(sentvec,Array(zeros(Float32,size(sentvec)[1],1)))
        end
    end
    return sentvec
end


function initlstmmodel(;rev=0)
    rnnSpec,rnnWeights = rnninit(EMBEDSIZE2,NUMHIDDEN2; rnnType=:lstm,dropout=DROPOUT)
    #inputMatrix = Array(xavier(Float32,EMBEDSIZE,MAXFEATURES))
    if rev ==0
        weightrange = sqrt(6/(OUTPUTSIZE2+NUMHIDDEN2*2))
        weightmat = weightrange.*rand(Float32,OUTPUTSIZE2,NUMHIDDEN2*2)
        #outputMatrix = convert(Array{Float32},weightrange.*xavier(Float32,OUTPUTSIZE2,NUMHIDDEN2*2))
        outputMatrix = convert(Array{Float32},weightmat-weightmat/2)
        outputBias = Array(zeros(Float32,OUTPUTSIZE2))
        weights = (rnnWeights,outputMatrix,outputBias)
    else
        weights = rnnWeights 
    end
    return (rnnSpec,weights)
end
function getrevvec(arr1)
    revar = arr1[:,1]
    for i=2:size(arr1,2)
        revar = hcat(arr1[:,i],revar)
    end
    return revar
end
##lstms = rspec1, rspec2  , weights = cnnweights,rnnweights, outputmatrix, rnnweights2,charembs,wordembs,capembs
function getbilstmoutput(lstms,weights,inputsent;bucketsize=30)
    input = KnetArray(getsentvec(weights[1],weights[end-1],weights[end],weights[end-2],inputsent[1],bucketsize))
    #input = reshape(input,(size(input)[1],size(input)[2],1))
    #for i =2:length(inputsent)
    #    input = cat(dims=3,input,KnetArray(getsentvec(weights[1],weights[end-1],weights[end],inputsent[i])))
    #end
    revinput = getrevvec(input)
    #println("inputs")
    #println(value(input))
    #println(value(revinput))
    #flush(stdout)
    flstmout = Array(rnnforw(lstms[1],weights[2],input)[1])
    blstmout = Array(rnnforw(lstms[2],weights[5],revinput)[1])
    outs = weights[3] * vcat(flstmout[:,1],blstmout[:,end]).+weights[4]
    if bucketsize ==1
        return outs
    end
    for i =2:bucketsize
        outs = hcat(outs,(weights[3]*vcat(flstmout[:,i],blstmout[:,end-i+1]).+weights[4]))
    end
    allouts = reshape(outs,(size(outs)[1],size(outs)[2],1))
    for j = 2:length(inputsent)
        input =KnetArray(getsentvec(weights[1],weights[end-1],weights[end],weights[end-2],inputsent[j],bucketsize))
        revinput = getrevvec(input)
        flstmout = Array(rnnforw(lstms[1],weights[2],input)[1])
        blstmout = Array(rnnforw(lstms[2],weights[5],revinput)[1])
        outs = weights[3] * vcat(flstmout[:,1],blstmout[:,end])+weights[4]
        #println(summary(outs))
        #println(summary(input))
        #println("girmeden")
        for i =2:bucketsize
            outs = hcat(outs,(weights[3]*vcat(flstmout[:,i],blstmout[:,end-i+1])+weights[4]))
        end
        allouts = cat(dims=3,allouts,outs)
    end
    #println("lstm out")
    #println(summary(allouts))
    #println(summary(inputsent))
    #flush(stdout)
    return allouts
end

function initmodel(pretrainfile)
    f = open(pretrainfile)
    lines = readlines(f)
    pretrains=Dict()
    for line in lines
        linesp = split(line)
        pretrains[linesp[1]] = map(x->parse(Float32,x),linesp[2:end])
    end
    charembs , wordembs ,capembs= initembs(CEMBDIM,WEMBDIM,CAPEMBDIM)
    println("before")
    println(wordembs[:,w2i["the"]])
    for key in keys(pretrains)
        wordembs[:,w2i[key]] = pretrains[key]
    end
    println("after")
    println(wordembs[:,w2i["the"]])
    flush(stdout)
    cnnweights = initcnnweights(FILTERSIZE,FILTERNUM,CNNOutputSize)
    flstm,flstmweights = initlstmmodel()
    blstm,blstmweights = initlstmmodel(rev=1)
    trrange = sqrt(6/(4+2*OUTPUTSIZE2))
    transitions = convert(Array{Float32},trrange.*xavier(Float32,OUTPUTSIZE2+2,OUTPUTSIZE2+2))
    #transitions = Array(sqrt.(rand(Float64,(OUTPUTSIZE2+2,OUTPUTSIZE2+2))./(2*(OUTPUTSIZE2+2))))
    #transitions = Array(zeros(OUTPUTSIZE2+2,OUTPUTSIZE2+2))
    #println(transitions)
    #flush(stdout)
    return charembs,wordembs,capembs,cnnweights,flstm,flstmweights,blstm,blstmweights,transitions
end


## not really using at the moment
function predict(weights,inputs,rnnSpecs)
    outs = getbilstmoutput(rnnSpecs,weights,inputs)
    return outs
end


##not used now
function normalize(scorematrix,dim)
    scores = exp.(scorematrix)
    sums = sum(exp.(scorematrix),dims = dim)
    if dim == 2
        for i in 1:size(scorematrix)[1]
            scores[i,:]/=sums[i]
        end
    else
        for i in 1:size(scorematrix)[2]
            scores[:,i]/=sums[i]
        end
    end
    return scores
end

##negative log-likelihood for transition probabilities
function transloss(transitions,batchtags)
    totnll=0
    for i =1:size(batchtags)[2]
        tags = batchtags[:,i]
        #println("Tags bilgisi $(summary(tags))")
        #flush(stdout)
        mynll = 0
        prevtag = size(transitions)[1]-1
        for tag in tags
            exptrans = exp.(transitions[prevtag,:])
            negloss = -log(exptrans[tag]/sum(exptrans))
            mynll = mynll + negloss
            prevtag = tag
        end
        exptrans = exp.(transitions[prevtag,:])
        negloss = -log(exptrans[end]/sum(exptrans))
        mynll = mynll + negloss
        mynll/=length(tags)
        totnll+=mynll
    end
    return totnll
end
function log_sum_exp(tag_scores)
    ind = argmax(tag_scores)
    arr= tag_scores
    #println(ind)
    max_score = tag_scores[ind]
    score = max_score
    max_arr = [max_score for i=1:length(tag_scores)]
    score+=log(sum(exp.(tag_scores-max_arr)))
    return score
end
function get_forward_score(mypreds,transitionscores1)
    #println("scores")
    #println(preds)
    #flush(stdout)
    #for_score = Array([-1.1e10 for i=1:size(preds)[1]+2])
    transitionscores = KnetArray(transitionscores1)
    preds = KnetArray(mypreds)
    for_score = Vector{Any}()
    for i=1:size(preds)[1]+2
        push!(for_score,-1.1e10)
    end
    for_score[end-1] = 0
    for i=1:size(preds)[2]
        tag_scores = Vector{Any}()
        for k=1:size(preds)[1]+2
            push!(tag_scores,-1.1e10)
        end
        #tag_scores = Array(Any,[-1.1e10 for i=1:size(preds)[1]+2])
        for j=1:size(preds)[1]
            score = Vector{Any}([preds[j,i] for i=1:length(for_score)])
            tag_score = score.+for_score.+transitionscores[:,j]
            #println(tag_score)
            log_1 = log_sum_exp(tag_score)
            #println("log: $(log_1)")
            #println(summary(log_1))
            #flush(stdout)
            tag_scores[j] = log_1
        end
        for_score = tag_scores
        #push!(for_score,-1.0e10)
        #push!(for_score,-1.0e10)
        #println(for_score)
        #flush(stdout)
    end
    for_score.+transitionscores[:,end]
    return log_sum_exp(for_score)
end
function get_gold_score(bilstmscores,transitionscores,golds)
    bilstmscore = KnetArray(bilstmscores)
    score = 0
    prev_tag = size(transitionscores)[2]-1
    #println(bilstmscore)
    #flush(stdout)
    for i=1:length(golds)
        new_tag = golds[i]
        score+= exp(bilstmscore[new_tag,i]+transitionscores[prev_tag,new_tag])
        prev_tag = new_tag
    end
    score+=transitionscores[prev_tag,end]
    println("gold score")
    println(log(score))
    flush(stdout)
    return log(score)
end
function get_crf_loss(weights,sents,tags,lstms)
    batchbilstmscores = getbilstmoutput(lstms,weights[1:end-1],sents)
    totalloss = 0
    for i=1:size(batchbilstmscores)[3]
        sentscores=batchbilstmscores[:,:,i]
        #println(size(value(weights[end])))
        #println(t2i)
        flush(stdout)
        fs=get_forward_score(value(sentscores),value(weights[end]))
        gs=get_gold_score(value(sentscores),value(weights[end]),tags[:,i])
        #println(fs)
        #println(gs)
        #println(sentscores)
        #flush(stdout)
        totalloss+= value(fs) - value(gs) 
    end
    println(value(totalloss))
    flush(stdout)
    return value(totalloss)
end
    
   
##used to find best tag sequence during prediction mode
function viterbidecode(tagscores,transitions)
    alltagpreds=[]
    tagprobs = tagscores
    transprobs = transitions
    bestscores = [tagprobs[:,1].+transprobs[end-1,1:end-2]]
    bestparents = []
    for i in 2:length(tagprobs[1,:])
        bestscore = []
        bestparent = []
        for j in 1:size(tagprobs)[1]
            tagprob = tagprobs[j,i].+transprobs[1:end-2,j].+bestscores[i-1]
            push!(bestscore,maximum(tagprob))
            push!(bestparent,argmax(tagprob))
            #println(argmax(tagprob))
        end
        #println(bestscore)
        push!(bestscores,bestscore)
        push!(bestparents,bestparent)
    end
    #println(transprobs[1:end-2,end])
    finalbest = bestscores[end].+transprobs[1:end-2,end]
    finalparent = argmax(finalbest)
    #push!(bestparents,finalparent)
    bestpath = [finalparent]
    for parent in reverse(bestparents)
        push!(bestpath,parent[finalparent])
        finalparent = parent[finalparent]
    end
    return bestscores,bestparents,reverse(bestpath)
end
#bestscores , bestparents , bestpath = viterbidecode(preds,transitions)

function my_preds(testset,lstms,weights,taglist)
    preds= []
    for sent in testset
        #println("mypred")
        #println(sent)
        #flush(stdout)
        bilstmscores = value(getbilstmoutput(lstms,weights[1:end-1],[sent],bucketsize=length(sent)))
        #println(summary(bilstmscores))
        #println("cumle")
        #println(sent)
        #flush(stdout)
        bestscores , bestparents , bestpath = viterbidecode(bilstmscores,weights[end])
        predarr= []
        for ind in bestpath
            push!(predarr,taglist[ind])
        end
        #println(bilstmscores[:,3])
        #flush(stdout)
        #for i=1:size(bilstmscores)[2]
        #    push!(predarr,(taglist[argmax(bilstmscores[:,i])]))
            #println(bilstmscores[:,i])
        #end
        push!(preds,predarr)
    	#push!(preds,bilstmpreds)
    end
    #println("bilstm preds")
    #println(preds)
    #flush(stdout)
    return preds
end
function pred_out(sents,preds,golds)
    totaltags = 0
    totalpreds = 0
    correct = 0
    t=0
    acc = 0.0
    pre = 0
    rec = 0
    println(summary(sents))
    for i in 1:length(sents)
        for j in 1:length(sents[i])
            pred = preds[i][j]
            g = golds[i][j]
            println(string(sents[i][j]," ",g," ",pred))
            if g ==pred
                acc+=1
            end
            if g!="O"
                if g == pred 
                    correct+=1
		    totalpreds+=1
                else
                    if pred!="O"
                        totalpreds+=1
                    end
                end
                totaltags+=1
            else
                if pred!="O"
                    totalpreds+=1
                end
            end
            t+=1
                
        end
        #println("\n")
    end
    println(string("Accuracy: ",acc/t))
    println(string("Precision: ",correct/totalpreds))
    println(string("Recall: ",correct/totaltags))
end

###READ DATASET AND INITIALIZE DICTIONARIES
dataset , taglist = read_with_labels(trainingfile);
t2i = tag2i(taglist)
OUTPUTSIZE2 = length(taglist)
docs = read_sents(trainingfile);
wordembfile = "../GloVe-1.2/vectors.txt"
i2w, w2i = vocab(docs,wordembfile);
i2c, c2i = chardict(i2w);
trainflag = 0
trainsents = []
traintags = []
bucket_sizes = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 140]
groupeddata = Dict()
for s in bucket_sizes
    groupeddata[s]=[]
end
for doc in dataset
	for sent in doc
        for size in bucket_sizes
            if length(sent[1])< size
                push!(groupeddata[size],sent)
                break
            end
        end
		push!(trainsents,sent[1])
		push!(traintags,sent[2])
	end
end
counts = zeros(length(taglist)+2,length(taglist)+2)
for senttags in traintags
    prevtag = length(taglist)+1
    for tag in senttags
        counts[prevtag,t2i[tag]]+=1
        prevtag = t2i[tag]
    end
    counts[prevtag,length(taglist)+2]+=1
end
function mapcounts(x)
    if x<3
        return -10.0
    else 
        return 0.0
    end
end
function findbucket(num)
    for i in bucket_sizes
        if num < i
            return i 
        end
    end
end
counts =map(mapcounts,counts)
#counts = log.(counts.+1.0e-10)./10
newtransitions = convert(Array{Float32},counts)
println("new transitions")
println(newtransitions)
flush(stdout)
## INITIALIZE WEIGHTS AND LOSS FUNCTION
charembs,wordembs,capembs,cnnweights,flstm,flstmweights,blstm,blstmweights,oldtransitions= initmodel(wordembfile);
#loss(weights,sent,tags,lstms)=nll(getbilstmoutput(lstms,weights[1:end-1],sent),tags)
loss(weights,sent,tags,lstms,bucketsize)=nll(getbilstmoutput(lstms,weights[1:end-1],sent,bucketsize=bucketsize),tags)+transloss(weights[end],tags)
#loss(weights,sents,tags,lstms)=get_crf_loss(weights,sents,tags,lstms)
##transitions weightsini cikardim getbilstme verilen #input deigsmis oldu weigths[1:end-1] yap transitions ekleyince
#+transloss(weights[end],tags)
lossgradient = grad(loss)
LR = 0.001
lr = 0.015          # Learning rate
BETA_1=0.9        # Adam optimization parameter
BETA_2=0.999      # Adam optimization parameter
EPS=1e-08         # Adam optimization parameter

##TRAIN
BATCHSIZE = 10
MAXSENTLEN = 30
EPOCHS = 50
savename = "knetmodel2.jld"
modelname = "denememodel2"
datanum = floor(Int,length(dataset)/100)
len1 = 0
#a
#test accuracyy
testset , testtaglist = read_with_labels("test.txt");
testlength = length(testset)
sents = []
tags = []
for doc in testset
    for sent in doc
        push!(sents,sent[1])
        push!(tags,sent[2])
    end
end
testdata = sents,tags
numbatches = sum([length(groupeddata[key]) for key in keys(groupeddata)])/BATCHSIZE
total = sum([length(groupeddata[key]) for key in keys(groupeddata)])
sizes = [length(groupeddata[i]) for i in bucket_sizes]
sums = [sum(sizes[1:i])/total for i=1:length(bucket_sizes)]
println(sums)
println(numbatches)
println(summary(groupeddata))
println("data sizes")
for b in bucket_sizes
    println(length(groupeddata[b]))
end
#flush(stdout)
data = minibatch(trainsents,traintags,BATCHSIZE)
println("batched data")
println(summary(data))
#summary(data[1,:])
flush(stdout)

function getbucketid(rand1)
    for i=1:length(sums)
        if rand1< sums[i]
            return i
        end
    end
    return -1
end
function getbatch(bucketid,batchsize)
    sents = []
    tags = []
    for i=1:batchsize
        rand1=rand(1:length(groupeddata[bucketid]))
        push!(sents,groupeddata[bucketid][rand1][1])
        push!(tags,groupeddata[bucketid][rand1][2])
    end
    return sents,tags
end
###

### suan 10luk batch ile calisiyorum ancak emin degilim dogru oldugundan ozellikle transloss kismi sikintili
### padding yaptigim yerlerden de loss hesapliyorum bu mantikli mi ogren
##TRAIN 
sent1,tags = dataset[1][1]
model = (cnnweights,flstmweights[1],flstmweights[2],flstmweights[3],blstmweights,capembs,charembs,wordembs,newtransitions)
model2 = (cnnweights,flstmweights[1],flstmweights[2],flstmweights[3],blstmweights,charembs,wordembs,newtransitions)##transitions taken out
#onehots = getonehots(tags,onehotdict)
optim = optimizers(model, Adam; lr=LR, beta1=BETA_1, beta2=BETA_2, eps=EPS)#
#optim = optimizers(model,Momentum,lr=0.015,gclip=5.0,gamma=0.9)
#taginds1 = getsenttaginds(tags[1],t2i)
println(t2i)
println(length(dataset))
flush(stdout)
#len1 = 1
#loss1 = loss(model,sent1,taginds1,(flstm,blstm))
#println(loss1)
#println(transloss(model[end],taginds1))
bucketsize = 10
for i in 1:EPOCHS
    for i=1:numbatches
        rand1 = rand()
        bucket_id = getbucketid(rand1)
        bucketsize = bucket_sizes[bucket_id]
        sents,tags = getbatch(bucketsize,BATCHSIZE)
        #println("sents")
        #println(tags)
        #sents, tags = batch1 
        #println(sents)
        #flush(stdout)
        #taginds = getsenttaginds(tags,t2i)
        
        taginds1 = getsenttaginds(tags[1],t2i,bucketsize)
        taginds1 = reshape(taginds1,(length(taginds1),1))
        for i =2:length(tags)
            taginds1 = cat(dims=2,taginds1,getsenttaginds(tags[i],t2i,bucketsize))
        end
        #println("tags")
        #println(summary(taginds1))
        #println(summary(sents))
        #flush(stdout)
        grads = lossgradient(model,sents,taginds1,(flstm,blstm),bucketsize)

        #println("grads ne alemde")
        #println(value(grads))
        #flush(stdout)
        #preds = my_preds([sents[1]],(flstm,blstm),model,taglist)
        #println(taginds1[:,1])
        #flush(stdout)
        #println(taglist)
        #println(taginds1[:,1])
        #println(tags[1])
        #println(preds)
        #println("transitions")
        #println(model[end])
        #flush(stdout)
        #println(sents)
        #println(taginds1)
        #println(summary(grads))
        #println(grads)
        #flush(stdout)
        #lr = 0.015/(1+0.05*i)
        update!(model, grads,optim)
        #println("transitions")
        #println(value(model[end]))
        #flush(stdout)
        #print("oldu mu gercekten")
    end
    #lr = 0.015/(1+i)
    #loss1 = loss(model,sent1,taginds1,(flstm,blstm))
    preds = my_preds(testdata[1],(flstm,blstm),model,taglist)
    println(string("EPOCH NUM: ",i))
    pred_out(testdata[1],preds,testdata[2])
    #println("learning rate")
    #println(lr)
    flush(stdout)
end
#println(loss(model,sent1,taginds1,(flstm,blstm)))
#println(transitions[end-1,:])

##save model
savemodel(savename,modelname,model)


for i in 1:EPOCHS
    for batch1 in data
        sents, tags = batch1 
        #println(sents)
        #flush(stdout)
        #taginds = getsenttaginds(tags,t2i)
        
        taginds1 = getsenttaginds(tags[1],t2i)
        taginds1 = reshape(taginds1,(length(taginds1),1))
        for i =2:length(tags)
            taginds1 = cat(dims=2,taginds1,getsenttaginds(tags[i],t2i))
        end
        grads = lossgradient(model,sents,taginds1,(flstm,blstm))
        update!(model, grads)
        #println("transitions")
        #println(value(model[end]))
        #flush(stdout)
        #print("oldu mu gercekten")
    end
    #lr = 0.015/(1+i)
    #loss1 = loss(model,sent1,taginds1,(flstm,blstm))
    preds = my_preds(testdata[1],(flstm,blstm),model,taglist)
    println(string("EPOCH NUM: ",i))
    pred_out(testdata[1],preds,testdata[2])
    #println("learning rate")
    #println(lr)
    flush(stdout)
end
#println(loss(model,sent1,taginds1,(flstm,blstm)))
#println(transitions[end-1,:])

##save model
savemodel(savename,modelname,model)

