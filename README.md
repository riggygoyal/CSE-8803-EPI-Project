# Unraveling the Nexus of Illnesses and Heatwaves: Predictive Modeling for Early Warning Systems

| Sai Nikhil Vangala         | Rigved Goyal        | Suchet Sapre  |
| ------------ | ------------- | ------------------ | 
| <img src="/CSE-8803-EPI-Project/SRC/WebPage_items/Nikhil.png" alt="Sai Nikhil Vangala" width="300"/> | <img src="/CSE-8803-EPI-Project/SRC/WebPage_items/Rigved.jpeg" alt="Rigved Goyal" width="300"/>    | <img src="/CSE-8803-EPI-Project/SRC/WebPage_items/Suchet.jpeg" alt="Suchet Sapre" width="300"/> | 

| Research Paper | Software Tar-Ball Files| Final Presentation| Github Repository |
| ------------ | ------------- | ------------------ | ------------------|
| Link | Link | Link | Link |

The alarming increase in the frequency and intensity of heat waves in recent years, fueled by the ongoing climate change, presents a substantial and urgent threat to public health worldwide. As global temperatures continue to rise, the need to predict, understand, and mitigate heat-related illnesses (HRIs) has become a paramount concern for scientists, policymakers, and healthcare professionals. This research project embarked on a multifaceted exploration of the intricate correlations between temperature and HRIs, with a primaryfocus on the Pacific Northwest region, a geographical area emblematic of the climate challenges faced by diverse communities. This study aimed to reveal the nuanced dimensions of heat vulnerability, which can vary significantly from place to place. The ultimate goal was to contribute to the development of early warning systems and strategies that safeguard vulnerable populations not only in the Pacific Northwest but also in other regions, addressing the adverse effects of rising temperatures and heat-related health problems on a global scale.

Two main tasks were addressed in this project. In the correlative task, we observed that region 4 displayed the strongest temperature and HRI correlation while region 10 displayed the weakest such correlation. With respect to the prediction of heat-related illnesses in the pacific northwest region, we obtained RMSE values of 102.0521, 57.46, and 87.387 for the ARIMA, XGBoost, and LSTM models, respectively. These results indicate that the XGBoost model performed the best. This could be because the XGBoost model was able to account for the larger deviations in HRIs for region 10 better than the LSTM model. For future work, we wish to expand upon the LSTM. Specifically, exploring transformer and ensemble-based models could help bolster accuracy further by incorporating attention and information from multiple classifiers. Secondly, we plan to increase this forecasting window to determine whether our models can accurately predict the number of HRIs farther into the future. Finally, our paper focuses on using only temperature data for each region. In the future, we would look into using more weather-related data such as wind speeds and humidity to inform our HRI forecasts. 
