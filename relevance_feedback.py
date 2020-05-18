import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
#from main import tf_idf
def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    #np.argsort(sim[:, i])[:10]
    alpha = 0.000001
    beta = -0.0009
    for iterations in range(3):
        for i in range(30):
            relevant = np.argsort(-sim[:, i])[:10]
            irrelevant = np.argsort(sim[:, i])[:10]
            relevance_matrix = np.zeros((1,10625))
            for j in range(len(relevant)):
                array = vec_docs[j].toarray()
                relevance_matrix = np.vstack((relevance_matrix, array[0]))
            irrelevance_matrix = np.zeros((1,10625))
            for j in range(len(irrelevant)):
                array = vec_docs[j].toarray()
                irrelevance_matrix = np.vstack((irrelevance_matrix, array[0]))
            summedRelevance = np.sum(relevance_matrix, axis = 0)
            summedIrrelevance = np.sum(irrelevance_matrix, axis = 0)

            summedIrrelevance = np.array([summedIrrelevance])
            summedIrrelevance.T

            summedRelevance = np.array([summedRelevance])
            summedRelevance.T

            np.true_divide(summedRelevance, 10)
            np.true_divide(summedIrrelevance, 10)

            summedRelevance = summedRelevance*alpha
            summedIrrelevance = summedIrrelevance*beta


            tempSparse = vec_queries[i].toarray()
            queryArray = tempSparse[0]

            result = np.add(queryArray,summedRelevance)
            queryArray = np.add(result,summedIrrelevance)

            vec_queries[i]=sparse.csr_matrix(queryArray)

            # vec_queries = sparse.csr_matrix(queryArray)
            sim = cosine_similarity(vec_docs,vec_queries)
    rf_sim = sim # change
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    alpha = 0.000001
    beta = -0.0009
    for iterations in range(3):
        for i in range(30):
            relevant = np.argsort(-sim[:, i])[:10]
            irrelevant = np.argsort(sim[:, i])[:10]
            relevance_matrix = np.zeros((1,10625))
            for j in range(len(relevant)):
                array = vec_docs[j].toarray()
                relevance_matrix = np.vstack((relevance_matrix, array[0]))
            irrelevance_matrix = np.zeros((1,10625))
            for j in range(len(irrelevant)):
                array = vec_docs[j].toarray()
                irrelevance_matrix = np.vstack((irrelevance_matrix, array[0]))
            summedRelevance = np.sum(relevance_matrix, axis = 0)
            summedIrrelevance = np.sum(irrelevance_matrix, axis = 0)

            summedIrrelevance = np.array([summedIrrelevance])
            summedIrrelevance.T

            summedRelevance = np.array([summedRelevance])
            summedRelevance.T

            np.true_divide(summedRelevance, 10)
            np.true_divide(summedIrrelevance, 10)

            summedRelevance = summedRelevance*alpha
            summedIrrelevance = summedIrrelevance*beta


            tempSparse = vec_queries[i].toarray()
            queryArray = tempSparse[0]

            result = np.add(queryArray,summedRelevance)
            queryArray = np.add(result,summedIrrelevance)

            vec_queries[i]=sparse.csr_matrix(queryArray)

            # vec_queries = sparse.csr_matrix(queryArray)
            sim = cosine_similarity(vec_docs,vec_queries)


            #Start query extension
            #relevant = np.argsort(-sim[:, i])[:5]

            topFiveArray = np.argsort(sim[:, i])[:5]
            # # vectorizer = TfidfVectorizer()
            # queriesTemp = queries
            # print(len(queriesTemp))
            # for i in range(5):
            #     temp_vector = tfidf_model.inverse_transform(vec_docs[topFiveArray[i]].toarray())
            #     queriesTemp.append(temp_vector)
            #     print(len(queriesTemp))
            # print(len(queriesTemp),len(queriesTemp[0]))
            # tem_queries = tfidf_model.transform(queriesTemp)

            # tempVec= vec_queries
            wordsArray = tfidf_model.get_feature_names()
            # print(queriesFromInverse)
            # for k in range(30):
            queriesFromInverse = tfidf_model.inverse_transform(vec_queries[i].toarray())
                # print((queriesFromInverse))
                # print((queriesFromInverse[0]))

                # print(type(queriesFromInverse))
                # break
            for j in range(5):
                queriesFromInverse[0] = np.append(queriesFromInverse[0],wordsArray[topFiveArray[j]])
                # print(queriesFromInverse)

            s = ""
            for j in range(len(queriesFromInverse[0])):
                s+= queriesFromInverse[0][j]+ " "
                # print(s) 
            s.strip()
            s = [s]
                # arr.append(s)
            tempTransformed = tfidf_model.transform(s)
            vec_queries[i]=sparse.csr_matrix(tempTransformed)

    rf_sim = sim  # change
    return rf_sim