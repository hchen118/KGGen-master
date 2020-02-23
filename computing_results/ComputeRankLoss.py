# ======================================================
#  Title:  computig the LRAP
#  Description: LRAP: Label Ranking Average Precision
#  Input: The notated results
#  Author: Hao Chen
#  Date:   Jan 15th 2019
# ======================================================
import numpy as np
from sklearn.metrics import label_ranking_loss, coverage_error,label_ranking_average_precision_score


# read head entity, tail entity and relation form datafile
# input: data file name
#       tem_flag: true, for text embedding file, false: graph embedding file
# output: the list of entity pairs and relations
def Load_data_file(data_file_name):
    y_true = []
    y_score = []
    with open(data_file_name, "r") as Ora_file:
        for line in Ora_file:

            try:
                label, score, D_score, head, tail, relation = line.strip().split("\t")
            except:
                break

            # try:
            # if label == "N" or label == "E":
            if label == "N":
                y_true.append(1)
            else:
                y_true.append(0)

            y_score.append(float(score))

    Ora_file.close()

    y_true_list = []
    y_true_list.append(y_true)
    y_score_list = []
    y_score_list.append(y_score)

    y_true_arr = np.array(y_true_list)
    y_score_arr = np.array(y_score_list)

    return y_true_arr, y_score_arr


#def averagenum
def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


#print all items in a list
def print_list(a_list):
    for item in a_list:
        print (item)


if __name__ == '__main__':

    #Load all the data
    file_name_base = "./new/PredictedEntityPairs_Notated"

    ranking_loss_list = []
    coverage_error_list = []
    lrap_list = []
    for i in range(0,10):
        if i == 0:
            file_name = file_name_base + ".txt"
        else:
            file_name = file_name_base + str(i+1)+ ".txt"
        y_true, y_score = Load_data_file(file_name)
        rank_loss = label_ranking_loss(y_true, y_score)
        cover_error = coverage_error(y_true, y_score)
        lrap = label_ranking_average_precision_score(y_true, y_score)
        # print(rank_loss)
        ranking_loss_list.append(rank_loss)
        coverage_error_list.append(cover_error)
        lrap_list.append(lrap)


    print("the ranking loss is:")
    print_list(ranking_loss_list)

    print("\nthe coverage error is")
    print_list(coverage_error_list)

    print("\nthe label ranking average precision score is")
    print_list(lrap_list)

    # print("the average ranking loss is: " + str(averagenum(ranking_loss_list)))

    # print("the average coverage error is: " + str(averagenum(coverage_error_list)))




