function tag2i(taglist)
    t2i= Dict()
    c =1
    for tag in taglist
        t2i[tag] = c
        c= c + 1
    end
    t2i
end
function getsenttaginds(tags,t2i)
    arr1 = Vector{Int64}()
    for tag in tags
        push!(arr1,t2i[tag])
    end
    arr1
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

function vocab(docs)
    vocab1=["OOV"]
    w2i = Dict{String,Int}("OOV"=>1)
    for doc in docs
        for sent in doc
            for word in sent
                w = String(word)
                if get(w2i,w,-1)==-1
                    push!(vocab1,word)
                    w2i[word] = length(vocab1)
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