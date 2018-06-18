import tensorflow as tf
import pandas as pd
import random

def load_data(file_path):
    print "Load dataset from " + file_path
    #NN_test_X = file_path + "/NN_test_X.csv"
    #NN_test_X_3statns = file_path + "/NN_test_X_3statns.csv"
    NN_test_X_1statn = file_path + "/NN_test_X_1statn.csv"

    NN_test_Y_ADO = file_path + "/NN_test_Y_ADO.csv"
    NN_test_Y_RPV = file_path + "/NN_test_Y_RPV.csv"
    NN_test_Y_RSS = file_path + "/NN_test_Y_RSS.csv"
    NN_test_Y_USC = file_path + "/NN_test_Y_USC.csv"
    NN_test_Y_depth = file_path + "/NN_test_Y_depth.csv"
    NN_test_Y_eqLoc = file_path + "/NN_test_Y_eqLoc.csv"
    NN_test_Y_magnitude = file_path + "/NN_test_Y_magnitude.csv"

    colnames = []
    for i in range(1):
        colnames.append("lat_" + str(i))
        colnames.append("long_" + str(i))
    for i in range(0):
        colnames.append("diff_" + str(i))
    for i in range(1):
        colnames.append("arrival_" + str(i))
    
    #X_10stations = pd.read_csv(NN_test_X, header=None, names=colnames)
    #X_3stations = pd.read_csv(NN_test_X_3statns, header=None, names=colnames)
    X_1station = pd.read_csv(NN_test_X_1statn, header=None, names=colnames)
    Y_ADO = pd.read_csv(NN_test_Y_ADO, header=None, names=["arrival_ADO"])
    Y_RPV = pd.read_csv(NN_test_Y_RPV, header=None, names=["arrival_RPV"])
    Y_RSS = pd.read_csv(NN_test_Y_RSS, header=None, names=["arrival_RSS"])
    Y_USC = pd.read_csv(NN_test_Y_USC, header=None, names=["arrival_USC"])
    Y_depth = pd.read_csv(NN_test_Y_depth, header=None, names=["depth"])
    Y_eqLoc = pd.read_csv(NN_test_Y_eqLoc, header=None, names=["lat_eq", "long_eq"])
    Y_mag = pd.read_csv(NN_test_Y_magnitude, header=None, names=["mag"])

    #dataset = pd.concat([X_10stations, Y_ADO, Y_RPV, Y_RSS, Y_USC, Y_eqLoc, Y_depth, Y_mag], axis=1)
    #dataset = pd.concat([X_3stations, Y_ADO, Y_RPV, Y_RSS, Y_USC, Y_eqLoc, Y_depth, Y_mag], axis=1)
    dataset = pd.concat([X_1station, Y_ADO, Y_RPV, Y_RSS, Y_USC, Y_eqLoc, Y_depth, Y_mag], axis=1)
    #pd.set_option('display.max_columns', 100)
    #print dataset.head(3)

    feature_count = 3
    label_count = 8

    return dataset, feature_count, label_count

def split_data(dataset, feature_count, label_count):
    train = dataset.head(int(len(dataset)*0.8))
    train_X = train.iloc[:,0:feature_count]
    train_Y = train.iloc[:,feature_count:feature_count+label_count]
    #pd.set_option('display.max_columns', 100)
    #print train.head(5)
    #print train_X.head(5)
    #print train_Y.head(5)
    #print len(train)

    test = dataset.drop(train.index)
    test_X = test.iloc[:,0:feature_count]
    test_Y = test.iloc[:,feature_count:feature_count+label_count]
    #pd.set_option('display.max_columns', 100)
    #print test.head(5)
    #print test_X.head(5)
    #print test_Y.head(5)
    #print len(test)

    return train_X, train_Y, test_X, test_Y

def build_model(units, feature_count, label_count):
    X = tf.placeholder(tf.float32)
    #print X

    input_layer = X
    input_layer = tf.reshape(input_layer, [1, feature_count])
    #input_layer = tf.Print(input_layer, [input_layer], summarize=feature_count)
    #print input_layer

    dense_layer = input_layer
    for i in units:
        dense_layer = tf.layers.dense(dense_layer, i)
    dense_layer = tf.layers.dense(dense_layer, label_count)
    #dense_layer = tf.Print(dense_layer, [dense_layer], summarize=label_count)
    #print dense_layer

    model = {
            "Y_ADO" : dense_layer[0,0],
            "Y_RPV" : dense_layer[0,1],
            "Y_RSS" : dense_layer[0,2],
            "Y_USC" : dense_layer[0,3],
            "Y_depth" : dense_layer[0,4],
            "Y_lat" : dense_layer[0,5],
            "Y_long" : dense_layer[0,6],
            "Y_mag" : dense_layer[0,7]
            }
    #print model

    Y = tf.placeholder(tf.float32)
    #print Y

    output_layer = Y
    output_layer = tf.reshape(output_layer, [1, label_count])
    output_layer = {
            "Y_ADO" : output_layer[0,0],
            "Y_RPV" : output_layer[0,1],
            "Y_RSS" : output_layer[0,2],
            "Y_USC" : output_layer[0,3],
            "Y_depth" : output_layer[0,4],
            "Y_lat" : output_layer[0,5],
            "Y_long" : output_layer[0,6],
            "Y_mag" : output_layer[0,7]
            }
    #print output_layer

    arrival_output = [output_layer["Y_ADO"], output_layer["Y_RPV"], output_layer["Y_RSS"], output_layer["Y_USC"]]
    arrival_predict = [model["Y_ADO"], model["Y_RPV"], model["Y_RSS"], model["Y_USC"]]
    arrival_loss = tf.losses.absolute_difference(arrival_output, arrival_predict)
    arrival_optimizer = tf.train.AdadeltaOptimizer(0.01).minimize(arrival_loss)

    center_output = [output_layer['Y_depth'], output_layer['Y_lat'],  output_layer['Y_long']]
    center_predict = [model['Y_depth'], model['Y_lat'], model['Y_long']]
    center_loss = tf.losses.mean_squared_error(center_output, center_predict)
    center_optimizer = tf.train.AdadeltaOptimizer(0.01).minimize(center_loss)

    mag_output = output_layer["Y_mag"]
    mag_predict = model["Y_mag"]
    mag_loss = tf.losses.absolute_difference(mag_output, mag_predict)
    mag_optimizer = tf.train.AdadeltaOptimizer(0.01).minimize(mag_loss)

    loss = [arrival_loss, center_loss, mag_loss]
    optimizer = tf.group(arrival_optimizer, center_optimizer, mag_optimizer)

    return model, X, Y, loss, optimizer

def main(argv):
    file_path = argv[1]

    dataset, feature_count, label_count = load_data(file_path)

    train_X, train_Y, test_X, test_Y = split_data(dataset, feature_count, label_count)

    model, X, Y, loss, optimizer = build_model([128 for i in range(40)], feature_count, label_count)

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    with tf.Session() as sess:
        print "Init"
        sess.run(init_global)
        sess.run(init_local)

        max_step = 50
        saver = tf.train.Saver(max_to_keep=max_step)

        for step in range(max_step):
            # train
            print "Training"
            arrival_loss_list = []
            center_loss_list = []
            mag_loss_list = []
            training_order = range(len(train_X))
            random.shuffle(training_order)
            for i in training_order:
                _, val = sess.run(
                        [optimizer, loss],
                        feed_dict = {
                            X : train_X.iloc[i],
                            Y : train_Y.iloc[i]
                            }
                        )
                arrival_loss_list.append(val[0])
                center_loss_list.append(val[1])
                mag_loss_list.append(val[2])
            output = "{},{},{},{}".format(
                    step,
                    sum(arrival_loss_list)/len(arrival_loss_list),
                    sum(center_loss_list)/len(center_loss_list),
                    sum(mag_loss_list)/len(mag_loss_list)
                    )
            print "Save model"
            saver.save(sess, "./earthquake_1/" + str(step) + ".ckpt")
            print output
            with open("history_1_train.csv", "a") as myfile:
                myfile.write(output+"\n")

            # test
            print "Test"
            for i in range(len(test_X)):
                val, predict = sess.run(
                        [loss, model],
                        feed_dict = {
                            X : test_X.iloc[i],
                            Y : test_Y.iloc[i]
                            }
                        )
                output = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
                        step,
                        val[0],
                        val[1],
                        val[2],
                        predict["Y_ADO"],
                        predict["Y_RPV"],
                        predict["Y_RSS"],
                        predict["Y_USC"],
                        predict["Y_depth"],
                        predict["Y_lat"],
                        predict["Y_long"],
                        predict["Y_mag"],
                        test_Y.iloc[i,0],
                        test_Y.iloc[i,1],
                        test_Y.iloc[i,2],
                        test_Y.iloc[i,3],
                        test_Y.iloc[i,4],
                        test_Y.iloc[i,5],
                        test_Y.iloc[i,6],
                        test_Y.iloc[i,7]
                        )
                print output
                with open("history_1_test.csv", "a") as myfile:
                    myfile.write(output+"\n")

if __name__ == "__main__":
    tf.app.run(main)
