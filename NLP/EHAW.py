def basic():
    model_path = '/content/Bone/boneage/weight/model.pt'
    tjnet_path = '/content/Bone/boneage/weight/tjnet24.h5'
    global tjnet
    tjnet = tf.models.load_model(tjnet_path, compile=False)
    global yolo
    yolo = torch.load(model_path, map_location='cpu')
    yolo.conf = 0.4
    return 

def data(path):
    data = pd.read_excel('/content/Sample/골연령측정자료220307.xlsx')
    data = data.dropna()
    data['Cage'] = data['Cage'].round(1)
    data['Patient #'] = data['Patient #'].astype(int).astype(str)
    patient_id = os.path.basename(path)[:-4]
    index = data[data['Patient #'] == patient_id].index[0]
    gender = data[(data['Patient #'] == patient_id)]['sex'][index]
    if 'F' in gender: 
        gender = 0
    else: gender = 1
    height = data[(data['Patient #'] == patient_id)]['촬영일키'][index]
    age = data[(data['Patient #'] == patient_id)]['Cage'][index]
    return patient_id, gender,height,age

def prediction(path,i):
    start = time.time()
    patient_id, gender, height, age = data(path)
    file_name = os.path.basename(path).replace('bmp','jpg')

    jpg_path = '/content/jpg/'
    if not os.path.exists(jpg_path):
        os.makedirs(jpg_path)

    processed_img = bone.Bone_extraction(path)


    cv2.imwrite(jpg_path+file_name,processed_img)

    crops, yoloimg, result = bone.yolo_crop_img(jpg_path+file_name,yolo)
    X = bone.out_crop_img(crops,gender)
    global prediction_BA
    prediction_BA = bone.predict_zscore(X, tjnet)
    prediction_BA = prediction_BA.round(2)
    lms_df = pd.read_csv('/content/Bone/boneage/data/height_df.csv')
    prediction_H = bone.Height_prediction(gender,prediction_BA,height,lms_df)

    diff = (prediction_BA - age).round(3)
    sec = time.time()-start
    times = str(datetime.timedelta(seconds=sec)).split(".")[0]
    print(times)
    csv = pd.DataFrame({'환자번호':patient_id,
                        '성별' : gender,
                        '촬영일자_나이':age,
                        '예측나이':prediction_BA,
                        '연령차이':diff,
                        '현재신장':height,
                        '예측신장':prediction_H,
                        '시간':times
                        },index=[i])


    if not os.path.exists('/content/Prediction.csv'):
        csv.to_csv('/content/Prediction.csv')
        print(f'save > {patient_id}')
    else:
        A = pd.read_csv('/content/Prediction.csv',index_col=0)
        A = A.append(csv)
        A.to_csv('/content/Prediction.csv')
        print(f'save > {patient_id}')