# Deformable-Sketch(Edge)-Detection-Network
Sketch detection network with deformable convolution.  
When the input image is given, then model outs it's sketch(edge) version.  
I use my custom handmake sketch(edge) dataset instead general benchmark datasets.  

## Test Envrionment  
- NVIDIA A5000 24G
- Train 512 x 512 image
- Model params: 6.2M

## Dataset  
[Anime] Dataset (included nude pictures)  
- train: 120 images, edges  
- val: 10 images, edges  

## Performance  

|Metric|ODS|OIS|LPIPS(edge)|LPIPS(image)|
|---|---|---|---|---|
|UAED|||||
|MuGE|||||
|DSDN|||||
  

## Results  
### Aru(BlueArchive)    
![0bea28087016b57f31978e164ceca03e493bd27b](https://github.com/user-attachments/assets/9d054eef-6cb1-4f49-b6b2-7dd99a208cc8)
![0bea28087016b57f31978e164ceca03e493bd27b png](https://github.com/user-attachments/assets/979aa8d0-c7cb-43c7-a74d-87320af50bab)  
![0dc453c239b447752a00695da65d5f7f1cd1a004](https://github.com/user-attachments/assets/f23c3595-288c-49db-8677-b93f5c29310a)
![0dc453c239b447752a00695da65d5f7f1cd1a004 png](https://github.com/user-attachments/assets/c3535d5f-d1f5-4ead-944b-17987caa5232)  
![0fca67f26e408849e21ade789408e16b0d4f08f7](https://github.com/user-attachments/assets/eeb7c98a-f9c9-46ca-a5ba-e055d2157d73)
![0fca67f26e408849e21ade789408e16b0d4f08f7 png](https://github.com/user-attachments/assets/99dbf33e-6c0b-41bc-9b61-3f0dd9e3768b)  
  
### Shiroko(BlueArchive)  
![0ff09e08a9089e3ab992b95d26c3e9a9815813b3](https://github.com/user-attachments/assets/78f57b6d-5ae7-484f-904d-03e4b9b41b8d)
![0ff09e08a9089e3ab992b95d26c3e9a9815813b3 png](https://github.com/user-attachments/assets/7167b935-4a8c-4a96-9cef-5d610f1ff12c)  
![3a5738b6b2f689bf0370b2decb495227b81e277b](https://github.com/user-attachments/assets/45b77471-2519-45d8-92b2-afd9ab3e983c)
![3a5738b6b2f689bf0370b2decb495227b81e277b png](https://github.com/user-attachments/assets/167403ec-5382-4a0f-8d5c-8a7f826de86f)  
![3ade832bf5ed9561c681adf1b039659a2ecf13be](https://github.com/user-attachments/assets/269bdb24-ba8a-4735-acf0-e0b7034890b2)
![3ade832bf5ed9561c681adf1b039659a2ecf13be png](https://github.com/user-attachments/assets/5658c75c-4f1a-4b99-9f31-6b79d6557e04)  
  
### Wakamo(BlueArchive)  
![0a3294741a8b98330f520a3e0f44b1ae7fb1993a](https://github.com/user-attachments/assets/59a9d70f-cc6e-4777-9df2-ccebbed14be6)
![0a3294741a8b98330f520a3e0f44b1ae7fb1993a png](https://github.com/user-attachments/assets/0bb2658f-f82a-4956-a51a-f6d3df2fbc0d)  
![0f5722138b086e6f28e3a354087d0e0692ea3a03](https://github.com/user-attachments/assets/5cf815ee-ecf9-4b97-8fa9-76955c29bbd3)
![0f5722138b086e6f28e3a354087d0e0692ea3a03 png](https://github.com/user-attachments/assets/500c32b9-8178-4d65-aabf-2d924ac058dc)  
![d84a452d25ac20b9cccdac007cc5f12ea349ba85](https://github.com/user-attachments/assets/c64bfbb0-d791-4684-8deb-02e665f8ff5c)
![d84a452d25ac20b9cccdac007cc5f12ea349ba85 png](https://github.com/user-attachments/assets/231dbc25-2ee9-48b0-a88d-e479c322ce62)  
  
### Surtr(Arknights)   
![6b7bb28262eebebffeb3941db1258c0f5366a7b3](https://github.com/user-attachments/assets/76957745-744d-4071-ab93-7d77d7077127)
![6b7bb28262eebebffeb3941db1258c0f5366a7b3 png](https://github.com/user-attachments/assets/b5252b6c-0651-4f0d-b98e-f96ce5d6fb2c)  
![5a165acbcb8ef52378282b515b24e9c277740ebf](https://github.com/user-attachments/assets/1f4e42c4-d726-449c-a60e-fcec7d04ef77)
![5a165acbcb8ef52378282b515b24e9c277740ebf png](https://github.com/user-attachments/assets/c258ecf7-7407-497c-b368-c788097a0099)  
![fcede69367108db9f91019b5ff0f8792656d2bc7](https://github.com/user-attachments/assets/2d9a84ce-20b9-4182-96da-f3e3253c0618)
![fcede69367108db9f91019b5ff0f8792656d2bc7 png](https://github.com/user-attachments/assets/6f67c64b-af10-4efd-84ad-c7c2c6ef5126)  
    
### Amiya(Arknights)    
![c08049bd745098b0869968caf059811fff9eb567](https://github.com/user-attachments/assets/0ba35a66-8409-4c63-ad31-d82a887168ae)
![c08049bd745098b0869968caf059811fff9eb567 png](https://github.com/user-attachments/assets/b24eb798-d771-4bfa-999c-71bff0b1a354)   
![fbd362241e3e84b52ccd40f70d627fcc9b5dbdd1](https://github.com/user-attachments/assets/9abe97ea-14f9-49fe-bb5e-3d73f5a38db5)
![fbd362241e3e84b52ccd40f70d627fcc9b5dbdd1 png](https://github.com/user-attachments/assets/c6936ede-0c1c-4ff7-b460-b01f68427599)   
![ffdc6bef27d1243e6aef64116429421a22591219](https://github.com/user-attachments/assets/f9c36442-0359-46ab-887b-00b116f27536)
![ffdc6bef27d1243e6aef64116429421a22591219 png](https://github.com/user-attachments/assets/cbe0d2e2-b89c-4105-8abd-309bbc21f00f)   
  
### Theresa(Arknights)  
![ad6114c4155a7cd5ea4dca11554b787e325e1801](https://github.com/user-attachments/assets/33de9448-f037-4377-9126-bf31f715ab21)
![ad6114c4155a7cd5ea4dca11554b787e325e1801 png](https://github.com/user-attachments/assets/aad67450-e316-45d6-9560-1de53ff6ad3d)  
![c8cca7fcb8cc00f8c799f361b9d4c9832e4d0801](https://github.com/user-attachments/assets/75a69996-0633-44b6-b666-5f04ed95cf6f)
![c8cca7fcb8cc00f8c799f361b9d4c9832e4d0801 png](https://github.com/user-attachments/assets/52cf63eb-4043-493e-a2d9-9cd481e8016c)  
![f187bff62fd3eb24bb1e0a6a0ecd813451978e5a](https://github.com/user-attachments/assets/36116512-a86c-47bf-814a-3b3cb43335d0)
![f187bff62fd3eb24bb1e0a6a0ecd813451978e5a png](https://github.com/user-attachments/assets/cd3693ae-5e06-4c14-9b81-01505ce87193)  
  
## Benchmark([UAED](https://github.com/ZhouCX117/UAED_MuGE), [MuGE](https://github.com/ZhouCX117/UAED_MuGE))  
### BSDS500
#### Humans or Animals  
- poor performance about humans or animals detection.(Noise sensitive & too many detect detail informations)
  
![376086](https://github.com/user-attachments/assets/ed7ec510-cc8d-4361-8974-90d2452b1593)  
![376086_uaed](https://github.com/user-attachments/assets/b99e3abc-206b-45a4-abc4-648a9be1c621)
![376086_muge](https://github.com/user-attachments/assets/2e3c2079-4f05-4c09-a29c-80fd575df80a)
![376086_sdn](https://github.com/user-attachments/assets/e9360d86-defc-41a9-b28c-8bf90c729510)
  
![198087](https://github.com/user-attachments/assets/163b9ebf-c316-4364-b687-46dd2a198b19)  
![198087_uaed](https://github.com/user-attachments/assets/8c6b69b5-1b2e-48a0-a7c2-4eaf12eb2283)
![198087_muge](https://github.com/user-attachments/assets/3d025bb1-8410-4315-b8d6-75862b2aba6a)
![198087_sdn](https://github.com/user-attachments/assets/116afba2-0057-4b0e-ba31-8cc5f4131e9a)  
  
#### Structures  
- good performance about structures than humans or animals detection.  
  
![48017](https://github.com/user-attachments/assets/c78ee709-d24a-48c1-9c2d-078e31dab9e0)  
![48017_uaed](https://github.com/user-attachments/assets/40f155f7-a002-4261-8ae4-c70a12d6f609)
![48017_muge](https://github.com/user-attachments/assets/91dd3181-7682-4b9c-b681-5516f08877da)
![48017_sdn](https://github.com/user-attachments/assets/38209f15-3365-4770-bc16-b701ac274867)  

![120093](https://github.com/user-attachments/assets/0ac08153-2dcb-4482-a555-9b929c9da8ab)  
![120093_uaed](https://github.com/user-attachments/assets/44a3e80c-481d-4b4a-bef2-d7aae96d51ab)
![120093_muge](https://github.com/user-attachments/assets/9d28099f-e7b0-4fb7-be87-562625de1abf)
![120093_sdn](https://github.com/user-attachments/assets/0a5a3d08-3426-4daa-91e4-2e0ef1eef289)  


#### Pictures(like above examples)  
- good performance about pictures than sota edge detection networks.
  
![6b7bb28262eebebffeb3941db1258c0f5366a7b3](https://github.com/user-attachments/assets/3faf75b7-00a6-449d-adfb-a3f2fe0a215d)  
![6b7bb28262eebebffeb3941db1258c0f5366a7b3_uaed](https://github.com/user-attachments/assets/09399a9c-1d35-4950-8091-16f787abfa7c)
![6b7bb28262eebebffeb3941db1258c0f5366a7b3_muge](https://github.com/user-attachments/assets/13ecf6a3-4464-4370-bc15-58fa9b3aa922)
![6b7bb28262eebebffeb3941db1258c0f5366a7b3 png](https://github.com/user-attachments/assets/8f3d6b53-765d-41ed-bb9f-8240c1e51b3c)  

![fbd362241e3e84b52ccd40f70d627fcc9b5dbdd1](https://github.com/user-attachments/assets/50f5a5ca-dece-4a9e-a56e-c94c21d5d1e3)  
![fbd362241e3e84b52ccd40f70d627fcc9b5dbdd1_uaed](https://github.com/user-attachments/assets/faa993bd-c180-4ee4-8fe4-ce9f69e8a8d6)
![fbd362241e3e84b52ccd40f70d627fcc9b5dbdd1_muge](https://github.com/user-attachments/assets/2b395dca-7304-49e9-8c60-6223effb3a34)
![fbd362241e3e84b52ccd40f70d627fcc9b5dbdd1 png](https://github.com/user-attachments/assets/d4122b59-323f-4899-b12b-01dce34f7ce1)  
  



  



