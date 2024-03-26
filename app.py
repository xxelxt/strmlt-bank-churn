import streamlit as st
import pandas as pd
from collections import Counter
import pickle
import base64

def load_model(file_path):
    with open(file_path, 'rb') as f:
        models, ordinal_encoder = pickle.load(f)
    return models, ordinal_encoder

def preprocess_data(input_data, ordinal_encoder):
    processed_data = input_data.copy()

    columns_to_drop = ['id', 'CustomerId', 'Surname']
    processed_data.drop(columns=columns_to_drop, inplace=True)

    processed_data[['Gender', 'Geography']] = ordinal_encoder.transform(processed_data[['Gender', 'Geography']])
    
    processed_data['AssetRatio'] = processed_data['Balance'] / processed_data['EstimatedSalary']
    processed_data['Interaction'] = processed_data['NumOfProducts'] * (processed_data['IsActiveMember'] + 0.2)
    
    return processed_data

def bank_churn_model(input_data, models):
    predictions = []
    probabilities = []

    for model in models:
        try:
            prediction = model.predict(input_data)
            probabilities.append(model.predict_proba(input_data)[:,1])
        except AttributeError:
            raise NotImplementedError("This model doesn't support predict_proba method.")
        predictions.extend(prediction)

    final_prediction = Counter(predictions).most_common(1)[0][0]
    final_probability = sum(probabilities) / len(models)

    return final_prediction, final_probability

def predict_from_uploaded_file(models, uploaded_file, ordinal_encoder):
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)

        st.write("### Dữ liệu được tải lên:")
        st.write(test_data)

        if st.button('Dự đoán'):
            preprocessed_test_data = preprocess_data(test_data, ordinal_encoder)

            st.write("### Dữ liệu sau khi xử lý:")
            st.write(preprocessed_test_data)

            predictions = []
            probabilities = []
            for model in models:
                prediction, probability = bank_churn_model(preprocessed_test_data, [model]) 
                predictions.append(prediction)
                probabilities.append(probability)

            final_probability = sum(probabilities) / len(models)

            test_data['Predicted Probability'] = final_probability 
            test_data['Label'] = test_data['Predicted Probability'].apply(lambda x: 'Yes' if x > 0.5 else 'No')

            st.write("### Kết quả dự đoán:")
            st.write(test_data)
            csv = test_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Tải xuống kết quả</a>'
            st.markdown(href, unsafe_allow_html=True)

def predict_from_user_input(models, ordinal_encoder):
    st.sidebar.header("Thay đổi dữ liệu để dự đoán:")

    credit_score = st.sidebar.slider("Điểm tín dụng", min_value=0, max_value=850, step=1, value=700)
    geography = st.sidebar.selectbox("Quốc gia", options=["France", "Spain", "Germany"])
    gender = st.sidebar.selectbox("Giới tính", options=["Male", "Female"])

    age = st.sidebar.slider("Tuổi", min_value=18, max_value=92, step=1, value=20)
    tenure = st.sidebar.slider("Số năm gắn bó với NH", min_value=0, max_value=10, step=1, value=5)
    balance = st.sidebar.slider("Số dư", min_value=0, max_value=300000, step=500, value=100000)

    num_of_products = st.sidebar.slider("Số SP của NH đã đăng ký", min_value=1, max_value=5, step=1, value=2)
    has_cr_card = st.sidebar.selectbox("Có thẻ tín dụng không?", options=["Có", "Không"])
    is_active_member = st.sidebar.selectbox("KH có thường xuyên hoạt động không?", options=["Có", "Không"])
    estimated_salary = st.sidebar.slider("Lương ước tính", min_value=10, max_value=300000, step=500, value=100000)

    gender_mapped, geography_mapped = ordinal_encoder.transform([[gender, geography]])[0]

    # geography_mapped = {'Pháp': 0, 'Tây Ban Nha': 1, 'Đức': 2}
    # gender_mapped = {'Nam': 0, 'Nữ': 1}

    is_active_member_mapped = {'Có': 1, 'Không': 0}
    has_cr_card_mapped = {'Có': 1, 'Không': 0}

    user_input = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography_mapped],
        'Gender': [gender_mapped],
        # 'Geography': [geography_mapped[[geography]]],
        # 'Gender': [gender_mapped[gender]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card_mapped[has_cr_card]],
        'IsActiveMember': [is_active_member_mapped[is_active_member]],
        'EstimatedSalary': [estimated_salary]
    })

    user_input['AssetRatio'] = user_input['Balance'] / user_input['EstimatedSalary']
    user_input['Interaction'] = user_input['NumOfProducts'] * (user_input['IsActiveMember'] + 0.2)

    final_prediction, final_probability = bank_churn_model(user_input, models)

    user_input['Predicted Probability'] = final_probability
    user_input['Label'] = user_input['Predicted Probability'].apply(lambda x: 'Yes' if x > 0.5 else 'No')

    st.write("### Kết quả dự đoán:")
    st.write(f"- Xác suất rời đi: {final_probability}")
    st.write(f"- Khả năng rời đi: {'Có' if final_probability > 0.5 else 'Không'}")
    st.write("### Dữ liệu đã nhập:")
    
    # Tạo một DataFrame để kết hợp tất cả các bản ghi
    all_inputs = pd.DataFrame()
    if 'user_inputs' not in st.session_state:
        st.session_state['user_inputs'] = []

    st.session_state['user_inputs'].append(user_input)

    for idx, df in enumerate(st.session_state['user_inputs'], start=1):
        all_inputs = pd.concat([all_inputs, df], ignore_index=True)

    st.write(all_inputs)

    csv = all_inputs.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_results.csv">Tải xuống kết quả</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    st.title('Dự đoán khả năng rời bỏ ngân hàng')

    models, ordinal_encoder = load_model('ensemble_model.pkl')

    option = st.selectbox("Lựa chọn", ["Tải lên file", "Tự nhập dữ liệu"])

    if option == "Tải lên file":
        uploaded_file = st.file_uploader("#### Tải lên file CSV", type=['csv'])
        if uploaded_file is not None:
            predict_from_uploaded_file(models, uploaded_file, ordinal_encoder)
    elif option == "Tự nhập dữ liệu":
        predict_from_user_input(models, ordinal_encoder)

if __name__ == '__main__':
    main()