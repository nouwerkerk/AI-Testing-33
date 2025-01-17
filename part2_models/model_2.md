# Hint
Subgroups to Test for Bias
    - women/men
    - single mothers
    - multiple children
    - migrants


# How to use model
## Loading dataset
df = pd.read_csv('data/investigation_train_large_checked.csv')

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['checked', 'Ja', 'Nee']),
    df['checked'].astype('int'),
    test_size=0.2,
    random_state=42
)

## Specify model
onnx_model_path = "model_2.onnx"
sess = rt.InferenceSession(onnx_model_path)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

## Run ONNX model predictions
onnx_predictions = sess.run(
    None, {input_name: X_test.values.astype('float32')}
)[0]