#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 8/30/20 8:49 PM

import math
import numpy as np


class MaxEntClassifier:
    def __init__(self, labels, num_features):
      self.labels = labels
      self.num_labels = len(labels)
      self.num_features = num_features
      self.weights = np.zeros([self.num_labels, self.num_features])
      self.best_loss = 0
      self.best_accuracy = 0
      self.t = 0
        
      ################################### TRAIN ####################################
    def train(self,
              train_dict,
              dev_dict=None,
              learning_rate=0.001,
              batch_size=64,
              num_iter=50,
              lambda_value = 1,
              max_extend_iter = 10,
              verbose=False):
      """TODO: train MaxEnt model with mini-batch stochastic gradient descent.
          Feel free to add more hyperparmaeters as necessary."""
      
      train_set = train_dict['instance']  
      dev_set = dev_dict['instance']  
      train_labels = train_dict['label']  
      dev_labels = dev_dict['label']  
    
      num_batch = math.ceil(len(train_set)/batch_size)
      extend_iter = 0
      best_weights = 0
      
        
      while True:
          start, end = 0, batch_size
          for j in range(num_batch):
              if end > len(train_set):
                  end = len(train_set)
      
              batch_label = train_labels[start:end]
              batch = train_set[start:end]
              
              gradient = self.compute_gradient(batch, batch_label, lambda_value)
              self.weights -= learning_rate*gradient
              
              
              start += batch_size
              end += batch_size
          
          train_accuracy = self.accuracy(self.classify(train_set), train_labels)    
          train_loss = self.neg_log_likelihood(train_set, train_labels)
          dev_accuracy = self.accuracy(self.classify(dev_set), dev_labels) 
          dev_loss = self.neg_log_likelihood(dev_set, dev_labels)
          
          if verbose == True:
              (print('iteration {0}: train_accuracy: {1:.4f}, train_loss: {2:.4f}, var_accuracy: {3:.4f}, var_loss: {4:.4f}'
                     .format(self.t, train_accuracy, train_loss, dev_accuracy, dev_loss)))

          
          if (dev_accuracy > self.best_accuracy):
              self.best_loss = dev_loss
              self.best_accuracy = dev_accuracy
              best_weights = self.weights
              extend_iter = 0
          
          else:
              if ((dev_loss - self.best_loss) < -0.001) & (extend_iter < max_extend_iter):
                  extend_iter += 1
              else:
                  break
              
          if self.t == num_iter:
              break
              
          self.t += 1
      
      self.weights = best_weights
    
    ################################## COMPUTE ###################################
    
    def unnorm_score(self, instance):
        unnorm_score = self.weights.dot(instance.T)
        return unnorm_score
   
    
    def expected_prob(self, instance, axis=0):
               
        unnorm_score = self.unnorm_score(instance)
        unnorm_score_sum = np.log(np.exp(unnorm_score).sum(axis=0))
        unnorm_score_sum = np.squeeze(np.asarray(unnorm_score_sum))
       
        prob = np.exp(unnorm_score - unnorm_score_sum)
        
        return prob
    
    def compute_gradient(self, instance, instance_label, lambda_value=-1):
        exp_prob = self.expected_prob(instance)
        exp_prob_sum = (exp_prob*instance)
        features_function = np.zeros((self.num_labels, self.num_features))
        
        for i in range(self.num_labels):
            features_function[i] = instance[instance_label == i].sum(axis=0)
        
        
        derivatives = -features_function + exp_prob_sum
        
        if lambda_value > 0:
            gradient = lambda_value*self.weights + derivatives
        
        else:
            gradient = self.weights + derivatives
        
        return gradient
    
    def neg_log_likelihood(self, instance, instance_label, lambda_value=-1):
        #log_posterior_sum = np.log(self.unnorm_score(instance).sum())
        log_posterior_sum = np.log(np.exp(self.unnorm_score(instance)).sum())
        pred_prob_sum = 0
        
        for i in range(self.num_labels):
            pred_prob_sum += self.weights[i].dot(instance[instance_label == i].T).sum()
            
        neg_log_likelihood = (-pred_prob_sum + log_posterior_sum)/len(instance_label)
        
        if lambda_value > 0:
            l2 = (np.linalg.norm(self.weights.ravel())^2)*(lambda_value/2)
            neg_log_likelihood = l2 + neg_log_likelihood
            
       
        return neg_log_likelihood
          
    
    def classify(self, instance):
        exp_prob = self.expected_prob(instance)
        pred = np.squeeze(np.asarray(exp_prob.argmax(axis=0)))
      
        return pred
      
    def accuracy(self, pred_array, actual_array):
        accuracy = (pred_array == actual_array).mean()    
        return accuracy
      
    

