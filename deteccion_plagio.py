# Naomi Anciola Calderón  -  A01750363

# CAMBIOS GLOBALES PENDIENTES

# escoger modo (1 a 1, luego 1 a n y luego n a m), etc
# agregar reglas regex para casos especificos
# visualizacion
# aseveracion / juicio final


# LIBRERIAS
import os # para navegar filepath y entradas
import re # expresiones regulares para pre-procesamiento
# from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise # para comparar
import spacy

nlp = spacy.load("en_core_web_md")  # make sure to use larger model!




# INTERFAZ
# print('que modo le gustaria utilizar?')
# modo = input('a) 1 con 1    b) 1 con m  c) n con m')
umbral_cos = float(input('valor que desea usar como umbral para la distancia de coseno (entre 0 y 1): '))
umbral_jac = float(input('valor que desea usar como umbral para la distancia de jaccard(entre ): '))
# ENTRADA DE TEXTO
os.chdir("ORIGINALES")
archivos_originales = [doc for doc in os.listdir() if doc.endswith(f'.txt')]
textos_originales = [open(File, errors="ignore").read() for File in archivos_originales]
# if modo == 'a':
#     print('se encontraron los siguientes archivos')
#     c = 0
#     for i in archivos_originales:
#         print(c,') ', i)
#         c += 1

#     indices_archivos = input('selecciona el numero (o numeros separados por espacios) correspondientes a los archivos que quieres comparar:')
#     indices_archivos = indices_archivos.split()
#     for i in len(indices_archivos):
#         indices_archivos[i] = int(indices_archivos[i])
#     print(indices_archivos)
#     for i in indices_archivos:
#         print(archivos_originales[int(i)])

os.chdir("..")
os.chdir("FinalTest")
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
    pattern = r"\b[a-zA-Z0-9']+\b"
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
indice_vec_arch = list(zip(archivos_originales + archivos_plagiados, vectores, textos_originales + textos_plagiados))

# r_text = ",".join(str(element) for element in [archivos_originales + archivos_plagiados])
# tokens = nlp(r_text)
# # for token1 in tokens:
# #     for token2 in tokens:
# #         print(token1.text, token2.text, token1.similarity(token2))



# COMPARACION / EVALUACION / SALIDA
# Cambios sugeridos:
# separar pasos de comparacion y evaluacion y salida
# hacer funciones diferentes para escribir en file o imprimir
# implementar comparaciones con locality sensitive hashing (o embedding) para que sea mas eficiente
# agregar longest common substring, semantic similarity, distancia jaccard como medidas de similitud

similitud_cos = lambda text1, text2: pairwise.cosine_similarity([text1, text2])
def similitud_jac(text1,text2):
   #text1 = modified_tokenize(text1)
   #text2 = modified_tokenize(text2)
   intersection_cardinality = len(set.intersection(*[set(text1), set(text2)]))
   union_cardinality = len(set.union(*[set(text1), set(text2)]))
   return intersection_cardinality/float(union_cardinality)




# f = open("reporte_2.txt", "a")
n=0
for j in range(110,140):
    for i in range(0,109):

        s_cos = similitud_cos(vectores[i],vectores[j])
        s_jac = similitud_jac(indice_vec_arch[i][2],indice_vec_arch[j][2])

        # print(indice_vec_arch[j][0])
        # print(indice_vec_arch[j][1])
        # print(indice_vec_arch[j][2])

        distancia_cos = 1 - s_cos [0][1]
        distancia_jac = 1 - s_jac
        
        if distancia_cos < umbral_cos or distancia_jac < umbral_jac:
            # f.write("\n")
            # f.write(indice_vec_arch[j][0]+ "\n")
            # f.write("###################" + "\n")
            print("\n")
            print(indice_vec_arch[j][0])
            print("###################")


            # f.write("candidato encontrado: "+ str(indice_vec_arch[i][0])+ "\n")
            # f.write("distancia coseno: "+ str(distancia_cos)+ "\n")
            # f.write("distancia jac: "+ str(distancia_jac,)+ "\n")

            print("candidato encontrado: "+ str(indice_vec_arch[i][0]))
            print("distancia coseno: "+ str(distancia_cos))
            print("distancia jac: "+ str(distancia_jac)+ "\n")

            if distancia_cos < distancia_jac:
                print('Tipo de plagio: cambio de tiempo o cambio de voz')
            elif distancia_jac < distancia_cos:
                print('Tipo de plagio: insertar, reemplazar o desordenar frases')

            porcentaje = int((1 - distancia_jac) * 100)
            print('El porcentaje plagiado de este documento fue de', porcentaje, '%')
            #print(tokensa)
            #print(tokensb)
            
            n +=1
print()
print('numero de candidatos encontrados: ',n)
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