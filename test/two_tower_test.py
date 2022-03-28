import re

import tensorflow as tf
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

flags = tf.app.flags
flags.DEFINE_string("model_dir", "../examples/model_output", "")
flags.DEFINE_string("predict_data", "../data/two_tower/predict_data", "")
flags.DEFINE_string("csv_header", "house_id,vector", "")
flags.DEFINE_string("output", "../data/two_tower/item_emb.csv", "")
FLAGS = flags.FLAGS

def supplyElems(elems, n, default=-1):
    c = len(elems)
    while c < n:
        elems.append(default)
        c = len(elems)

def get_model_path(model_dir):
    root, dirs, files = os.walk(model_dir)

    return dirs[0]

import csv
def save_item_emb(info_ids, item_embs, output):
    csv_header = FLAGS.csv_header
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f, delimiter="\t")
        for info_id, item_emb in zip(info_ids, item_embs):
            item_emb_str = ','.join([str(x) for x in item_emb])
            info_id_str = str(info_id)
            writer.writerow([info_id_str, str(item_emb_str)])

def main():
    model_dir = get_model_path(FLAGS.model_dir)
    print("model_dir: ", model_dir)
    predict_fn = tf.contrib.predictor.from_saved_model(model_dir)
    predict_data_dir = FLAGS.predict_data
    examples = []
    info_ids = []
    for filename in os.listdir(predict_data_dir):
        file_path = os.path.join(predict_data_dir, filename)
        if not re.search('DS', filename) and os.path.isfile(file_path):
            print("file_path: ",file_path)
            with open(file_path, 'r', encoding='utf8') as f:
                for line in f.readlines():
                    elems = line.split('\001', -1)
                    ctr_label = [int(elems[4])]
                    cvr_label = [int(elems[5])]

                    supplyElems(user_area_index, 5, -1)

                    continuous_ft_pro = [float(elems[i]) for i in range(13, 26)]
                    continuous_ft_pro += [float(elems[i]) for i in range(28, 48)]
                    # continuous_ft_user = [float(elems[i]) for i in range(48, 58)]
                    continuous_ft_user = [-1 for i in range(48, 58)]

                    example = tf.train.Example(features=tf.train.Features(feature={
                        "info_id": tf.train.Feature(int64_list=tf.train.Int64List(value=info_id)),
                        "city_index": tf.train.Feature(int64_list=tf.train.Int64List(value=city_index)),
                        "region_index": tf.train.Feature(int64_list=tf.train.Int64List(value=region_index)),
                        "shangquan_index": tf.train.Feature(int64_list=tf.train.Int64List(value=shangquan_index)),
                        "comm_index": tf.train.Feature(int64_list=tf.train.Int64List(value=comm_index)),
                        "price_index": tf.train.Feature(int64_list=tf.train.Int64List(value=price_index)),
                        "area_index": tf.train.Feature(int64_list=tf.train.Int64List(value=area_index)),

                        "user_region_index": tf.train.Feature(int64_list=tf.train.Int64List(value=user_region_index)),
                        "user_shangquan_index": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=user_shangquan_index)),
                        "user_comm_index": tf.train.Feature(int64_list=tf.train.Int64List(value=user_comm_index)),
                        "user_price_index": tf.train.Feature(int64_list=tf.train.Int64List(value=user_price_index)),
                        "user_area_index": tf.train.Feature(int64_list=tf.train.Int64List(value=user_area_index)),

                        "continuous_feature_pro": tf.train.Feature(
                            float_list=tf.train.FloatList(value=continuous_ft_pro)),
                        "continuous_feature_user": tf.train.Feature(
                            float_list=tf.train.FloatList(value=continuous_ft_user)),

                        "cvr_label": tf.train.Feature(int64_list=tf.train.Int64List(value=cvr_label)),
                        # "ctr_label": tf.train.Feature(int64_list=tf.train.Int64List(value=ctr_label))
                    }))
                    info_ids.append(info_id[0])
                    examples.append(example.SerializeToString())
    #开始预测
    predictions = predict_fn({'examples': examples})
    item_embs = predictions['item_net']
    # print(item_embs)
    save_item_emb(info_ids, item_embs, FLAGS.output)

if __name__ == '__main__':
    main()
    # get_model_path(FLAGS.model_dir)

