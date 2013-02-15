# Visualization of training.
# 
# Author: tombr
###############################################################################

require(ggplot2)

dataset = data.frame();

if (F) {
  
  # Shared bias only
  newlog = data.frame(read.csv("logs/sharedbias.csv"), test.type = "Shared bias");
  newlog = data.frame(batch = 1:nrow(newlog), newlog);
  dataset = rbind(dataset, newlog);
  
  # Bias only
  newlog = data.frame(read.csv("logs/onlybias.csv"), test.type = "Only bias");
  newlog = data.frame(batch = 1:nrow(newlog), newlog);
  dataset = rbind(dataset, newlog);
  
  # Weights and bias terms
  newlog = data.frame(read.csv("logs/weightsandbias.csv"), test.type = "Weights and bias");
  newlog = data.frame(batch = 1:nrow(newlog), newlog);
  dataset = rbind(dataset, newlog); 
}

### Weights and bias

# Layer1
newlog = data.frame(read.csv("logs/l1_p4_f32_e600.csv"), layer = "Layer 1", test.type = "Weights and bias");
newlog = data.frame(batch = 1:nrow(newlog), newlog);
dataset = rbind(dataset, newlog);

# Layer2
newlog = data.frame(read.csv("logs/l2_p2_f32_e750.csv"), layer = "Layer 2", test.type = "Weights and bias");
newlog = data.frame(batch = 1:nrow(newlog), newlog);
dataset = rbind(dataset, newlog);

# Layer3
newlog = data.frame(read.csv("logs/l3_p2_f64_e1000.csv"), layer = "Layer 3", test.type = "Weights and bias");
newlog = data.frame(batch = 1:nrow(newlog), newlog);
dataset = rbind(dataset, newlog);

# Layer4
newlog = data.frame(read.csv("logs/l4_p2_f64_e1000.csv"), layer = "Layer 4", test.type = "Weights and bias");
newlog = data.frame(batch = 1:nrow(newlog), newlog);
#dataset = rbind(dataset, newlog);

### SHARED

# Layer1
newlog = data.frame(read.csv("logs/shared_l1_p4_f32_e1000.csv"), layer = "Layer 1", test.type = "Shared");
newlog = data.frame(batch = 1:nrow(newlog), newlog);
dataset = rbind(dataset, newlog);

# Layer2
newlog = data.frame(read.csv("logs/shared_l2_p2_f32_e1000.csv"), layer = "Layer 2", test.type = "Shared");
newlog = data.frame(batch = 1:nrow(newlog), newlog);
dataset = rbind(dataset, newlog);

# Layer3
newlog = data.frame(read.csv("logs/shared_l3_p2_f64_e1000.csv"), layer = "Layer 3", test.type = "Shared");
newlog = data.frame(batch = 1:nrow(newlog), newlog);
dataset = rbind(dataset, newlog);

myplot = ggplot(dataset, aes(x = batch)) +
    
    # Hidden unit activations
    geom_smooth(aes(y = h.mean, colour = "Hidden activation", fill = "Hidden activation"), alpha = 0.5) +
    
    # Reconstruction
    geom_line(aes(y = vneg.mean, colour = "Reconstruction", fill = "Reconstruction"), alpha = 0.5) +
    
    # Error
    geom_smooth(aes(y = error, colour = "Error", fill = "Error")) +
    
    # Weights
    #geom_line(aes(y = F.mean, colour = "Weights", fill = "Weights")) +
    #geom_ribbon(aes(ymin = F.mean - 2 * F.sd, ymax = F.mean + 2 * F.sd, fill = "Weights"), alpha = 0.2) +
    
    # Visible bias
    geom_line(aes(y = b.mean, colour = "Visible bias", fill = "Visible bias")) +
    #geom_ribbon(aes(ymin = b.mean - 2 * b.sd, ymax = b.mean + 2 * b.sd, fill = "Visible bias"), alpha = 0.2) +
    
    # Hidden bias
    #geom_line(aes(y = c.mean, colour = "Hidden bias", fill = "Hidden bias")) +
    #geom_ribbon(aes(ymin = c.mean - 2 * c.sd, ymax = c.mean + 2 * c.sd, fill = "Hidden bias"), alpha = 0.2) +
    
    facet_grid(layer ~ test.type, scale = "free_x") +
    #facet_wrap(~ test.type, ncol = 2, scales = "free_x") +
    
    # Labels and theme
    labs(x = NULL, y = NULL, colour = "Quantity", fill = "Quantity");
print(myplot);
