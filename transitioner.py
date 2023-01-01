def Transitioner(input_list):
    predict_list = []
    if input_list[0] == 'Male':
        predict_list.append(1)
    elif input_list[0] == 'Female':
        predict_list.append(0)
    if input_list[1] == 'White':
        predict_list.append(0)
        predict_list.append(0)
        predict_list.append(1)
    elif input_list[1] == 'Hispanic':
        predict_list.append(1)
        predict_list.append(0)
        predict_list.append(0)
    elif input_list[1] == 'Black':
        predict_list.append(0)
        predict_list.append(0)
        predict_list.append(0)
    else:
        predict_list.append(0)
        predict_list.append(1)
        predict_list.append(0)
    if (int(input_list[2]) >= 30) and (int(input_list[2]) <= 44):
        predict_list.append(1)
        predict_list.append(0)
        predict_list.append(0)
    elif (int(input_list[2]) >= 45) and (int(input_list[2]) <= 59):
        predict_list.append(0)
        predict_list.append(1)
        predict_list.append(0)
    elif int(input_list[2]) >= 60:
        predict_list.append(0)
        predict_list.append(0)
        predict_list.append(1)
    elif (int(input_list[2]) >= 18) and (int(input_list[2]) <= 29):
        predict_list.append(0)
        predict_list.append(0)
        predict_list.append(0)
    if input_list[3] == 'No college':
        predict_list.append(1)
        predict_list.append(0)
    elif input_list[3] == 'Postgraduate':
        predict_list.append(0)
        predict_list.append(1)
    elif input_list[3] == 'Bachelors':
        predict_list.append(0)
        predict_list.append(0)
    if int(input_list[4]) >= 200000:
        predict_list.append(1)
        predict_list.append(0)
        predict_list.append(0)
        predict_list.append(0)
    elif (int(input_list[4]) >= 30000) and (int(input_list[4]) < 50000):
        predict_list.append(0)
        predict_list.append(1)
        predict_list.append(0)
        predict_list.append(0)
    elif (int(input_list[4]) >= 50000) and (int(input_list[4]) < 100000):
        predict_list.append(0)
        predict_list.append(0)
        predict_list.append(1)
        predict_list.append(0)
    elif int(input_list[4]) < 30000:
        predict_list.append(0)
        predict_list.append(0)
        predict_list.append(0)
        predict_list.append(1)
    elif (int(input_list[4]) >= 100000) and (int(input_list[4]) < 200000):
        predict_list.append(0)
        predict_list.append(0)
        predict_list.append(0)
        predict_list.append(0)
    if input_list[5] == 'Unmarried':
        predict_list.append(1)
    elif input_list[5] == 'Married':
        predict_list.append(0)
    return predict_list
