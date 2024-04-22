# Naomi Anciola Calderón  -  A01750363

# CAMBIOS GLOBALES PENDIENTES
# implementar typing
# programar interfaz que permita interactuar con los parametros dinamicamente, 
# escoger modo (1 a 1, luego 1 a n y luego n a m), etc
# agregar reglas regex para casos especificos



# LIBRERIAS
import os # para navegar filepath y entradas
import re # expresiones regulares para pre-procesamiento
# from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import pairwise # para comparar


# ENTRADA DE TEXTO
os.chdir("ORIGINALES")
archivos_originales = [doc for doc in os.listdir() if doc.endswith(f'.txt')]
textos_originales = [open(File, errors="ignore").read() for File in archivos_originales]
os.chdir("..")
os.chdir("PLAGIADOS")
archivos_plagiados = [doc for doc in os.listdir() if doc.endswith(f'.txt')]
textos_plagiados = [open(File, errors="ignore").read() for File in archivos_plagiados]
os.chdir("..")


# TOKENIZACION
# Cambios sugeridos:    
# Modificar patron de regex para que no excluya numeros
# guardar tokens (y texto) en zip de (nombre de archivo, texto, tokens, vector) 
# para usarlo en distancia jaccard

vector_tokens = []
def modified_tokenize(text):
    # exclude numbers, underscores, and punctuation
    pattern = r"\b[a-zA-Z']+\b"
    text = text.lower()
    words = re.findall(pattern, text)
    # print("Words found by regex:", words)  # Debug print

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    # print("Lemmatized tokens: ", lemmatized)

    # stemmer = PorterStemmer()
    # stemmed = [stemmer.stem(word) for word in tokens]
    # print("Stemmed tokens: ", stemmed)

    tokens = sorted(set(lemmatized))

    # vector_tokens.append(tokens) # algo asi para guardar token en vector
    # print("Unique tokens:", tokens)  # Debug print
    return tokens


# VECTORIZACION
# Cambios sugeridos:
# intentar con metodos de vectorizacion alterntivos (spacy word2vec, etc)

vectorizer = lambda Text: TfidfVectorizer(
    analyzer="word", 
    stop_words="english", 
    strip_accents = "unicode",
    tokenizer=modified_tokenize,
    token_pattern=None
    ).fit_transform(Text).toarray()

vectores = vectorizer(textos_originales + textos_plagiados)
indice_vec_arch = list(zip(archivos_originales + archivos_plagiados, vectores))


# COMPARACION / EVALUACION / SALIDA
# Cambios sugeridos:
# separar pasos de comparacion y evaluacion y salida
# hacer funciones diferentes para escribir en file o imprimir
# implementar comparaciones con locality sensitive hashing (o embedding) para que sea mas eficiente
# agregar longest common substring, semantic similarity, distancia jaccard como medidas de similitud

similitud_cos = lambda text1, text2: pairwise.cosine_similarity([text1, text2])
# def similitud_jac(text1,text2):
#   intersection_cardinality = len(set.intersection(*[set(text1), set(text2)]))
#   union_cardinality = len(set.union(*[set(text1), set(text2)]))
#   return intersection_cardinality/float(union_cardinality)

# f = open("reporte_2.txt", "a")
for j in range(110,120):
    # f.write("\n")
    # f.write(indice_vec_arch[j][0]+ "\n")
    # f.write("###################" + "\n")
    print("\n")
    print(indice_vec_arch[j][0])
    print("###################")
    for i in range(0,109):
        s_cos = similitud_cos(vectores[i],vectores[j])
        #s_jac = similitud_jac(indice_vec_arch[j][1])
        # print(indice_vec_arch[j][0])
        # print(indice_vec_arch[j][1])
        # print(indice_vec_arch[j][2])
        distancia_cos = 1 - s_cos [0][1]
        # distancia_jac = 1 - s_jac
        
        if distancia_cos < 0.8: # or distancia_jac < 0.8:
            # f.write("candidato encontrado: "+ str(indice_vec_arch[i][0])+ "\n")
            # f.write("distancia coseno: "+ str(distancia_cos)+ "\n")
            # f.write("distancia jac: "+ str(distancia_jac,)+ "\n")
            print("candidato encontrado: "+ str(indice_vec_arch[i][0]))
            print("distancia coseno: "+ str(distancia_cos))
            #print("distancia jac: "+ str(distancia_jac,)+ "\n")
            #print(tokensa)
            #print(tokensb)
        
# f.close()
# f = open("reporte.txt", "r")
# print(f.read()) 


# BIBLOGRAFIA
# Ya utilizados:
# https://www.geeksforgeeks.org/plagiarism-detection-using-python/
# https://perso.crans.org/besson/publis/notebooks/My_own_tiny_similarit..

# Por utilizar:
# https://www.geeksforgeeks.org/continuous-bag-of-words-cbow-in-nlp/
# https://spotintelligence.com/2023/01/16/local-sensitive-hashing-lsh/
# https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/
# https://v2.spacy.io/usage/vectors-similarity
# https://v2.spacy.io/api/annotation#pos-tagging
# https://v2.spacy.io/usage/linguistic-features#dependency-parse
# https://v2.spacy.io/usage/rule-based-matching