import random

import numpy as np

import torch
import torch.nn.functional as F


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Construct new tensor that requires gradient computation
    X = X.clone().detach().requires_grad_(True)

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores, and then compute the gradients with torch.autograd.grad.           #
    ##############################################################################

    # compute the gradient of the unnormalized score corresponding to the correct class
    # (which is a scalar) with respect to the pixels of the image
    # Forward pass to compute the scores and loss
    scores = model(X)
    scores = scores.gather(1, y.view(-1, 1)).squeeze()
                # scores 텐서에서 특정 인덱스에 위치한 값을 선택. 실제 레이블(y)
                # gather(1, ...): 1은 차원을 의미
                # y.view(-1, 1): y 텐서 (N,) 형태. 이를 (N,1) 형태로 재구성
                # squeeze(): 크기가 1인 차원을 제거. (N,1)을 (N,)으로 재구성

    # Backward pass to compute the gradient of the correct class score 
    # with respect to each input image
    model.zero_grad()   # 모델의 모든 parameter에 대해 누적된 gradient를 0으로 초기화
    scores.backward(torch.ones_like(scores))    
                # 'scores' 텐서에 대한 gradient 계산
                # 각 입력 텐서에 대한 gradient를 누적 합
                # torch.ones_like(scores): 역전파를 시작하는 데 필요한 gradient 초기값
    # 일반적으로 손실함수에서 backward()를 호출할 때, 손실이 스칼라 값이기 때문에 인자가 없지만,
    # 'scores'와 같이 스칼라가 아닌 경우 gradient 초기값을 명시적으로 제시해야 함.
    # 단, 샐리언스 맵을 계산할 때는 각 클래스 점수에 대한 gradient를 계산하는 것이 목적이므로,
    # 각 점수의 변화가 입력 이미지에 어떻게 영향을 미치는 지를 파악.

    # To compute the saliency map, take the absolute value of this gradient,
    # then take the maximum value over the 3 input channels. shape (H, W)
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
                # X.grad: scores.backward() 함수에서 계산된 입력 텐서 X에 대한 gradient
                # .data: gradient 텐서의 데이터 부분에 접근. gradient 자체의 값
                # .abs(): gradient의 절대값 계산. 즉, gradient 크기
                
                # torch.max(..., dim=1): 
                            # 텐서의 최대값과 인덱스 반환.
                            # 'dim=1'은 최대값을 계산할 차원. 입력 텐서 X가 (N,C,H,W)일 때,
                            # C차원에 대해 최대값을 계산. 즉, 모든 채널의 각 픽셀 위치에서 최대값
    
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image.
    X_fooling = X.clone().detach().requires_grad_(True)

    learning_rate = 1
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. You should perform gradient ascent on the score of the #
    # target class, stopping when the model is fooled.                           #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################

    # gradient ascent를 이용해 X_fooling 이미지를 조금씩 조정하여,
    # 모델이 이 이미지를 target_y 클래스로 분류하도록 유도하는 것이 목표
    for i in range(100):
        # Forward pass
        scores = model(X_fooling)
                # 모델의 출력으로, 각 클래스에 대한 점수를 담고 있는 (N,C) 형태의 텐서
                # N은 배치 크기, C는 클래스 수

        # 이미지가 fooling image와 같아질 때 stop
        if scores.argmax().item() == target_y:
                # scores 텐서에서 가장 높은 점수를 가진 인덱스를 찾는다
                # .item(): argmax가 반환하는 텐서에서 단일 값을 파이썬 기본 데이터 타입으로 변환
            break

        # The score of the target class
        target_scores = scores[0, target_y]
                # scores[0, ...]: 첫 번째 차원에서 0번째 요소. N=1이므로, 첫번째 사진
                # 단일 스칼라 값, 특정 이미지(X_fooling)가 특정 클래스(target_y)로 분류될 때 모델의 예측 점수
        
        # Perform gradient ascent on the score
        model.zero_grad()           # Initialize gradient
        target_scores.backward()    
                # target_scores가 의존하는 모든 텐서(이 경우 X_fooling)에 대한 그래디언트를 계산하고, 
                # 이를 해당 텐서의 .grad 속성에 저장합니다.
                # 이 gradient는 target_y 클래스의 점수를 최대화하기 위해 X_fooling을 어떻게 변경해야 하는지를 나타냄
        gradient = X_fooling.grad.data

        # Normalize the gradient and update the fooling image
        dX = learning_rate * gradient / gradient.norm() # 정규화
        X_fooling.data += dX    # 정규화된 gradient(dX)를 update하여 target_y 클래스로 분류하도록
        X_fooling.grad.data.zero_() # gradient 초기화

        # 진행 상황 출력
        print(f"반복 {i + 1}, 목표 점수: {target_scores.item()}")

    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling.detach()


def update_class_visualization(model, target_y, l2_reg, learning_rate, img):
    """
    Perform one step of update on a image to maximize the score of target_y
    under a pretrained model.

    Inputs:
    - model: A pretrained CNN that will be used to generate the image
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - img: the image tensor (1, C, H, W) to start from
    """

    # Create a copy of image tensor with gradient support
    img = img.clone().detach().requires_grad_(True)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    ########################################################################

    # this is just like the make_fooling_image, but only one step...
    y_pred = model(img)
        
    # loss function is the score of our desired class
    score = y_pred[0, target_y] + l2_reg * img.norm()
    score.backward()

    grad = img.grad
    dX = learning_rate * grad / grad.norm()

    # now ADD the gradient to get closer to target_y!
    with torch.no_grad():
        img += dX
        # X_fooling.grad.zero_()

    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img.detach()
