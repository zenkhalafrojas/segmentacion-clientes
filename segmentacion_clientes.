# =============================================================================
# Segmentacion de clientes del Mall Customers Dataset
# Modelos: K-Means y clustering jerarquico (ward.D), evaluados con la silueta
# =============================================================================

# Paquetes necesarios
# install.packages(c("ggplot2", "cluster"))
library(ggplot2)
library(cluster)

# -----------------------------------------------------------------------------
# 1. Carga del dataset (Mall_Customers.csv esta junto a este script)
# -----------------------------------------------------------------------------
clientes <- read.csv("Mall_Customers.csv")
head(clientes)

# -----------------------------------------------------------------------------
# 2. Exploracion y limpieza de datos
# -----------------------------------------------------------------------------
dim(clientes)      # 200 clientes y 5 variables
str(clientes)      # tipos de cada columna
summary(clientes)  # resumen estadistico

# Comprobamos que no hay valores nulos ni clientes duplicados
colSums(is.na(clientes))
sum(duplicated(clientes$CustomerID))

# Renombramos las columnas para facilitar su manejo
# (read.csv convierte "Annual Income (k$)" en "Annual.Income..k..", etc.)
names(clientes) <- c("CustomerID", "Genre", "Age", "Annual_Income", "Spending_Score")
head(clientes)

# Codificamos Genre como variable numerica: 1 = masculino, 0 = femenino
clientes$Genre <- ifelse(clientes$Genre == "Male", 1, 0)
table(clientes$Genre)  # numero de hombres (1) y mujeres (0)

# Normalizamos las variables numericas clave para que sean comparables.
# Annual_Income va de 15 a 137 (k$) y Spending_Score de 1 a 100: sin escalar,
# la variable con mayor rango dominaria el calculo de distancias.
# scale() las transforma a media 0 y desviacion tipica 1.
datos_clustering <- scale(clientes[, c("Annual_Income", "Spending_Score")])
summary(datos_clustering)

# Nota: CustomerID es solo un identificador y no se usa para segmentar.
# La segmentacion se basa en ingresos y gasto; la edad y el genero se usan
# despues para describir el perfil de cada segmento.

# -----------------------------------------------------------------------------
# 3. Exploracion de variables
# -----------------------------------------------------------------------------
# Distribucion de la edad
ggplot(clientes, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "steelblue", color = "white") +
  labs(title = "Distribucion de la edad", x = "Edad", y = "Numero de clientes") +
  theme_minimal()

# Distribucion por genero
ggplot(clientes, aes(x = factor(Genre, labels = c("Femenino", "Masculino")))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Clientes por genero", x = "Genero", y = "Numero de clientes") +
  theme_minimal()

# Distribucion de los ingresos anuales
ggplot(clientes, aes(x = Annual_Income)) +
  geom_histogram(binwidth = 10, fill = "steelblue", color = "white") +
  labs(title = "Distribucion de los ingresos anuales",
       x = "Ingresos anuales (k$)", y = "Numero de clientes") +
  theme_minimal()

# Distribucion de la puntuacion de gasto
ggplot(clientes, aes(x = Spending_Score)) +
  geom_histogram(binwidth = 10, fill = "steelblue", color = "white") +
  labs(title = "Distribucion de la puntuacion de gasto",
       x = "Puntuacion de gasto (1-100)", y = "Numero de clientes") +
  theme_minimal()

# Diagramas de caja para detectar valores extremos
boxplot(clientes[, c("Age", "Annual_Income", "Spending_Score")],
        main = "Diagramas de caja de las variables numericas",
        names = c("Edad", "Ingresos (k$)", "Gasto (1-100)"),
        col = "lightblue")

# Relacion entre ingresos y gasto: ya se intuyen varios grupos de clientes
ggplot(clientes, aes(x = Annual_Income, y = Spending_Score)) +
  geom_point(color = "steelblue", size = 2) +
  labs(title = "Ingresos anuales frente a puntuacion de gasto",
       x = "Ingresos anuales (k$)", y = "Puntuacion de gasto (1-100)") +
  theme_minimal()

# Observaciones de la exploracion:
# - No hay valores nulos ni duplicados.
# - Hay mas mujeres que hombres.
# - La edad se concentra entre los 20 y los 50 anos.
# - Los ingresos muestran un unico valor alto aislado (137 k$), que no es un
#   error sino un cliente real de ingresos muy altos, por lo que se mantiene.
# - El grafico de ingresos frente a gasto muestra a simple vista unos cinco
#   grupos: uno central y cuatro en las esquinas.

# -----------------------------------------------------------------------------
# 4. Entrenamiento de modelos de clustering
# -----------------------------------------------------------------------------

## 4.1 K-Means ----------------------------------------------------------------
set.seed(123)  # para que los resultados sean reproducibles

# Suma de cuadrados interna (tot.withinss) para k = 2..6
valores_k <- 2:6
wss <- sapply(valores_k, function(k) {
  kmeans(datos_clustering, centers = k, nstart = 25)$tot.withinss
})
data.frame(k = valores_k, tot.withinss = round(wss, 2))

# Metodo del codo
ggplot(data.frame(k = valores_k, wss = wss), aes(x = k, y = wss)) +
  geom_line(color = "steelblue") +
  geom_point(color = "steelblue", size = 3) +
  labs(title = "Metodo del codo (K-Means)",
       x = "Numero de clusters (k)",
       y = "Suma de cuadrados interna (tot.withinss)") +
  theme_minimal()

# El codo aparece en k = 5: hasta ahi la suma de cuadrados interna cae con
# fuerza y, a partir de 5, la mejora al anadir un cluster mas es pequena.
k_optimo <- 5

# Entrenamos el modelo final con k = 5 y asignamos los clusters al dataset
set.seed(123)
modelo_kmeans <- kmeans(datos_clustering, centers = k_optimo, nstart = 25)
clientes$cluster_kmeans <- factor(modelo_kmeans$cluster)
table(clientes$cluster_kmeans)  # tamano de cada segmento

## 4.2 Clustering jerarquico --------------------------------------------------
# Matriz de distancias euclidianas sobre los datos normalizados
distancias <- dist(datos_clustering, method = "euclidean")

# Clustering jerarquico con el metodo de Ward
modelo_jerarquico <- hclust(distancias, method = "ward.D")

# Dendrograma
plot(modelo_jerarquico, labels = FALSE, hang = -1,
     main = "Dendrograma (metodo ward.D, distancia euclidiana)",
     xlab = "Clientes", ylab = "Altura", sub = "")
rect.hclust(modelo_jerarquico, k = 5, border = 2:6)  # marca los 5 grupos

# Cortamos el dendrograma en 5 clusters (mismo k que en K-Means, para poder
# compararlos en igualdad de condiciones) y asignamos los clusters al dataset
clientes$cluster_jerarquico <- factor(cutree(modelo_jerarquico, k = 5))
table(clientes$cluster_jerarquico)  # tamano de cada segmento

# Comparacion de las asignaciones de ambos modelos
table(KMeans = clientes$cluster_kmeans, Jerarquico = clientes$cluster_jerarquico)

# -----------------------------------------------------------------------------
# 5. Evaluacion de modelos
# -----------------------------------------------------------------------------
# Silueta de K-Means
silueta_kmeans <- silhouette(modelo_kmeans$cluster, distancias)
media_silueta_kmeans <- mean(silueta_kmeans[, "sil_width"])

# Silueta del clustering jerarquico
silueta_jerarquico <- silhouette(cutree(modelo_jerarquico, k = 5), distancias)
media_silueta_jerarquico <- mean(silueta_jerarquico[, "sil_width"])

# Comparacion
comparacion <- data.frame(
  Modelo = c("K-Means", "Jerarquico (ward.D)"),
  Silueta_media = round(c(media_silueta_kmeans, media_silueta_jerarquico), 4)
)
comparacion

# Graficos de silueta de cada modelo
plot(silueta_kmeans, col = 2:6, border = NA, main = "Silueta - K-Means (k = 5)")
plot(silueta_jerarquico, col = 2:6, border = NA, main = "Silueta - Jerarquico (k = 5)")

# Seleccion del mejor modelo: el de mayor silueta media
mejor_modelo <- comparacion$Modelo[which.max(comparacion$Silueta_media)]
cat("Mejor modelo segun la silueta media:", mejor_modelo, "\n")

# Resultados:
# - K-Means: silueta media de 0,5547.
# - Jerarquico (ward.D): silueta media de 0,5538.
# Ambos valores (cercanos a 0,55) indican segmentos razonablemente compactos y
# bien separados. Los dos modelos son practicamente equivalentes: coinciden en
# la asignacion de 196 de los 200 clientes (solo 4 clientes de la frontera
# entre el grupo central y el de ingresos altos/gasto bajo cambian de grupo).
# Seleccionamos K-Means por tener la silueta media ligeramente superior.

# -----------------------------------------------------------------------------
# 6. Analisis descriptivo de segmentos
# -----------------------------------------------------------------------------
# Perfil medio de cada segmento de K-Means
perfil_kmeans <- aggregate(cbind(Age, Annual_Income, Spending_Score, Genre) ~ cluster_kmeans,
                           data = clientes, FUN = mean)
perfil_kmeans$n_clientes <- as.vector(table(clientes$cluster_kmeans))
names(perfil_kmeans)[names(perfil_kmeans) == "Genre"] <- "Pct_hombres"
perfil_kmeans[, -1] <- round(perfil_kmeans[, -1], 2)
perfil_kmeans

# Perfil medio de cada segmento del clustering jerarquico
perfil_jerarquico <- aggregate(cbind(Age, Annual_Income, Spending_Score, Genre) ~ cluster_jerarquico,
                               data = clientes, FUN = mean)
perfil_jerarquico$n_clientes <- as.vector(table(clientes$cluster_jerarquico))
names(perfil_jerarquico)[names(perfil_jerarquico) == "Genre"] <- "Pct_hombres"
perfil_jerarquico[, -1] <- round(perfil_jerarquico[, -1], 2)
perfil_jerarquico

# Nota: como Genre vale 1 para hombres y 0 para mujeres, su media es la
# proporcion de hombres de cada segmento (columna Pct_hombres).

# Asignamos a cada segmento un nombre descriptivo segun sus ingresos y su
# gasto medios, para que la interpretacion no dependa del numero del cluster
nombrar_segmento <- function(ingresos, gasto) {
  nivel_ingresos <- ifelse(ingresos < 40, "ingresos bajos",
                    ifelse(ingresos > 70, "ingresos altos", "ingresos medios"))
  nivel_gasto <- ifelse(gasto < 35, "gasto bajo",
                 ifelse(gasto > 65, "gasto alto", "gasto medio"))
  paste0(toupper(substr(nivel_ingresos, 1, 1)), substring(nivel_ingresos, 2),
         ", ", nivel_gasto)
}
perfil_kmeans$Segmento <- nombrar_segmento(perfil_kmeans$Annual_Income,
                                           perfil_kmeans$Spending_Score)
perfil_kmeans[order(-perfil_kmeans$n_clientes), ]

# INTERPRETACION DE LOS PERFILES (modelo seleccionado: K-Means, k = 5)
#
# 1) Ingresos medios, gasto medio (81 clientes; edad media ~43; ~55 k$; gasto ~50)
#    Es el segmento mas numeroso: el cliente "estandar" del centro comercial.
#    Estrategia: promociones generales y programas de fidelizacion para
#    aumentar la frecuencia de compra.
#
# 2) Ingresos altos, gasto alto (39 clientes; edad media ~33; ~87 k$; gasto ~82)
#    Clientes jovenes-adultos con alto poder adquisitivo que ya gastan mucho.
#    Es el segmento mas valioso. Estrategia: programa VIP, lanzamientos
#    exclusivos y atencion personalizada para retenerlos.
#
# 3) Ingresos altos, gasto bajo (35 clientes; edad media ~41; ~88 k$; gasto ~17)
#    Tienen capacidad de compra pero gastan poco: es el segmento con mas
#    potencial desaprovechado. Es el unico con mayoria de hombres (~54 %).
#    Estrategia: investigar por que no compran (oferta poco atractiva,
#    preferencia por otros canales) y probar productos premium o experiencias.
#
# 4) Ingresos bajos, gasto bajo (23 clientes; edad media ~45; ~26 k$; gasto ~21)
#    Clientes de mas edad y prudentes, sensibles al precio.
#    Estrategia: descuentos, ofertas y productos de buena relacion calidad-precio.
#
# 5) Ingresos bajos, gasto alto (22 clientes; edad media ~25; ~26 k$; gasto ~79)
#    Los clientes mas jovenes: gastan mucho pese a sus ingresos bajos.
#    Estrategia: moda y tendencias, redes sociales y promociones puntuales,
#    vigilando que el gasto sea sostenible (p. ej., con cuidado al ofrecer
#    financiacion).
#
# Patrones principales:
# - Los ingresos no explican por si solos el gasto: con ingresos altos hay un
#   grupo que gasta mucho y otro que gasta muy poco.
# - La edad diferencia el comportamiento: los segmentos que mas gastan son los
#   mas jovenes (medias de ~25 y ~33 anos) y los que menos gastan, los mayores
#   (~41-45 anos).
# - El genero esta bastante equilibrado entre segmentos, asi que no es un
#   factor diferenciador relevante en esta segmentacion.

# -----------------------------------------------------------------------------
# 7. Visualizacion
# -----------------------------------------------------------------------------
# Clusters de K-Means, con los centroides en las unidades originales
centroides <- aggregate(cbind(Annual_Income, Spending_Score) ~ cluster_kmeans,
                        data = clientes, FUN = mean)

ggplot(clientes, aes(x = Annual_Income, y = Spending_Score, color = cluster_kmeans)) +
  geom_point(size = 2.5, alpha = 0.8) +
  geom_point(data = centroides, aes(x = Annual_Income, y = Spending_Score),
             color = "black", shape = 4, size = 5, stroke = 2) +
  labs(title = "Segmentos de clientes - K-Means (k = 5)",
       subtitle = "Las cruces negras marcan los centroides",
       x = "Ingresos anuales (k$)", y = "Puntuacion de gasto (1-100)",
       color = "Cluster") +
  theme_minimal()

# Clusters del modelo jerarquico
ggplot(clientes, aes(x = Annual_Income, y = Spending_Score, color = cluster_jerarquico)) +
  geom_point(size = 2.5, alpha = 0.8) +
  labs(title = "Segmentos de clientes - Clustering jerarquico (ward.D, k = 5)",
       x = "Ingresos anuales (k$)", y = "Puntuacion de gasto (1-100)",
       color = "Cluster") +
  theme_minimal()

# Edad frente a gasto, coloreando por los segmentos de K-Means
ggplot(clientes, aes(x = Age, y = Spending_Score, color = cluster_kmeans)) +
  geom_point(size = 2.5, alpha = 0.8) +
  labs(title = "Edad frente a puntuacion de gasto por segmento (K-Means)",
       x = "Edad", y = "Puntuacion de gasto (1-100)", color = "Cluster") +
  theme_minimal()

# Ingresos de cada segmento (K-Means)
ggplot(clientes, aes(x = cluster_kmeans, y = Annual_Income, fill = cluster_kmeans)) +
  geom_boxplot(show.legend = FALSE) +
  labs(title = "Ingresos anuales por segmento (K-Means)",
       x = "Cluster", y = "Ingresos anuales (k$)") +
  theme_minimal()
