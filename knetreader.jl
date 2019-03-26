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
            push!(tags,split(line)[2])
            if !(split(line)[2] in taglist)
                push!(taglist,split(line)[2])
            end
        end
    end
    docs,taglist
end

#docs,taglist = read_with_labels("knetfolder/train.txt");
#taglist