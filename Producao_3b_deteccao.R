library(lidR)
library(lidaRtRee)
library(itcSegment)
library(raster)
library(geosphere)
library (tidyverse)
library(sf)
library(PiC)
library(ForestTools)
library(ggpubr)
library(GGally)
library(yardstick)
library(ggplot2)
library(ggExtra)
library(cowplot)

setwd('C:\\Users\\danielsilva\\Meu Drive\\Projetos ativos\\Produção de erva-mate por visão computacional\\Dados\\Dados 3 Barras')
output_folder <- 'C:\\Users\\danielsilva\\Meu Drive\\Projetos ativos\\Produção de erva-mate por visão computacional\\Dados\\Dados 3 Barras\\arvores\\detec'

nv <- lidR::readLAS("nv_rgb_pre_normalizada.las")
plot(nv)

arvs <- st_read("D:\\erva_mate\\deep-forest\\arvs_deepforest_refined_10.shp") #polígonos das plantas individuais 
arvs <- tibble::rowid_to_column(arvs, "ID")

st_crs(arvs) == st_crs(nv)

# Recorta nuvem e aplica modelo pra estimar produção #### 
unique <- unique(arvs$ID)
arvores <- list()
producao <- data.frame(ID=character(), prod=numeric())

for (i in 1:length(unique)) {
  # seleciona a arvore 
  s <- arvs[arvs$ID == unique[i], ]
  # registra o ID da árvore
  id <- s$ID
  # mostra o ID da árvore
  print(id)
  # cria o nome para o novo objeto
  nome <- paste0('arv_',unique[i]) 
  # recorta a nuvem de acordo com o shape
  n <- clip_roi(nv, s)
  # Verifica se a nuvem recortada está vazia ou nula
  if (is.null(n) || is.empty(n)) {
    prod <- 0
  } else {
    # Remove pontos de chão se a nuvem não for vazia
    n <- remove_ground(n)
    
    # Se a remoção de chão deixar a nuvem vazia, também atribui 0
    if (is.empty(n)) {
      prod <- 0
    } else {
      res <- 0.1
      voxel <- voxel_metrics(n, ~list(occ = TRUE), res = res, all_voxels = FALSE)
      vox_vol <- res^3 
      v_v <- nrow(voxel) * vox_vol 
      
      prod <- 0.5976 + 5.0331 * v_v
    }
  }
  
  # Salva a linha no dataframe final
  tmp <- data.frame(ID = nome, prod = prod, stringsAsFactors = FALSE)
  producao <- bind_rows(producao, tmp)
}


df_final <- bind_cols(arvs, producao)

st_write(obj = df_final, dsn = "D:\\erva_mate\\deep-forest\\arvs_deepforest_refined_10_prod.shp")



# Gráfico da produção 

ggplot(data = producao, aes(y = prod, x = 1)) +
  geom_jitter(height = 0, width = 0.3, alpha = 0.3, size = 6, color = 'darkgreen') +
  geom_violin(color = 'darkgreen', fill = 'darkgreen', alpha = 0.5) + 
  theme_classic() +
  labs(
    x = NULL,
    y = "Yield (kg)"
  ) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.text.y = element_text(size = 15),
        axis.title.y = element_text(size = 15),
        axis.ticks.x=element_blank()
     )


plyr::count(producao$prod)
sum(producao$prod)
a <- filter(producao, producao$prod > 0)
mean(a$prod)
max(producao$prod)
