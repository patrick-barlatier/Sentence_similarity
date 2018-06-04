// Auteur : Hugo Delile


//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
/*
Programme qui calcul la distance entre tous les sens de deux mots.
Le fichier en entrée est la sortie du programe Distributed Multi-sense Word Embedding (fichier binary embedding)

Ici en format texte :

64627 76525 200									=> Première valeur : le nombre de mot, deuxième valeur : nombre de sens, dernière valeur : taille du vecteur.
</s> 2											=> Nombre de sens du mot.
1.0000 -0.265 -0.073 [...] -0.067 -0.368		=> Première valeur : probabilité, le reste : vecteur correspondant au mot.
0.0000 0.051 -0.039 [...] -0.072 0.172
le 2
0.0000 0.048 -0.127 [...] -0.086 0.102
1.0000 0.130 0.049 [...] -0.170 0.247
[...]


*/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include "sparsepp.h"

using namespace spp;

const long long MAX = 10;
const long long max_size = 2000;         // max length of strings
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
	FILE*																fi;

	char																st1[max_size];
	char																file_name[max_size];
	char																st[max_size];
	
	int																	nombreDeSynonyme;
	int																	sens;

	float																bestd[MAX];
	float																dist;
	float																prob;
	float																len;
	float																vec_1[max_size];

	long long															words;
	long long															size;
	long long															a;
	long long															b;
	long long															c;
	long long															d;
	long long															e;
	long long															bi;
	long long															bestw[MAX];

	float *																M;
	float *																probas;
	float *																probas_sel;
	float *																dists;

	char *																vocab;
	char *																mot;
	char *																motTemp;

	sparse_hash_map<std::string, long long>								mots;
	sparse_hash_map<long long, std::string>								id_mots;
	sparse_hash_map<std::string, long long>::iterator					iterateur;
	sparse_hash_map<long long, int>										id_sens;

	if (argc < 3) {
		printf("utilisation: ./distance <FICHIER>\n où FICHIER contient les mots sous forme de vecteur au format bianire\n");
		return 0;
	}
	strcpy(file_name, argv[1]);

	nombreDeSynonyme = atoi(argv[2]);
	
	fi = fopen(file_name, "rb");
	if (fi == NULL) {
		printf("Fichier en entrée non trouvé\n");
		return -1;
	}
	fscanf(fi, "%lld", &words);
	fscanf(fi, "%lld", &size); //Nombre de sens total en tout inutile.
	fscanf(fi, "%lld", &size);

	vocab											= (char *)malloc((long long)words * max_w * sizeof(char));
	mot												= (char *)malloc(max_w * sizeof(char));
	motTemp											= (char *)malloc(max_w * sizeof(char));

	M												= (float *)malloc((long long)nombreDeSynonyme * (long long)words * (long long)size * sizeof(float));
	probas											= (float *)malloc((long long)nombreDeSynonyme * (long long)words * sizeof(float));
	probas_sel										= (float *)malloc((long long)nombreDeSynonyme * (long long)nombreDeSynonyme * sizeof(float));
	dists											= (float *)malloc((long long)nombreDeSynonyme * (long long)nombreDeSynonyme * sizeof(float));


	if (M == NULL) {
		printf("Dépassement de la mémoire : %lld MB    %lld  %lld\n", (long long)nombreDeSynonyme * (long long)words * (long long)size * sizeof(float) / 1048576, words, size);
		return -1;
	}

	for (b = 0; b < words; b++) {
		a = 0;
		while (1) {
			mot[a] = fgetc(fi);

			if (feof(fi) || (mot[a] == ' ')) break;
			if ((a < max_w) && (mot[a] != '\n')) a++;
		}

		mot[a] = 0;
		strcpy(motTemp, mot);
		mots.emplace(motTemp, b);
		id_mots.emplace(b, motTemp);
		a = 0;
		while (1) {
			mot[a] = fgetc(fi);

			if (feof(fi) || (mot[a] == ' ')) break;
			if ((a < max_w) && (mot[a] != '\n')) a++;
		}

		mot[a] = 0;
		sens = atoi(mot);
		id_sens.emplace(b, sens);

		for (c = 0; c < sens; c++)
		{
			fread(&probas[((b * nombreDeSynonyme) + c)], sizeof(float), 1, fi);
			for (a = 0; a < size; a++) fread(&M[a + size * ((b * nombreDeSynonyme) + c)], sizeof(float), 1, fi);
			len = 0;
			for (a = 0; a < size; a++) len += M[a + size * ((b * nombreDeSynonyme) + c)] * M[a + size * ((b * nombreDeSynonyme) + c)];
			len = sqrt(len);
			for (a = 0; a < size; a++) M[a + size * ((b * nombreDeSynonyme) + c)] /= len;

		}
	}
	fclose(fi);
	
	while (1) {

		printf("Entrer un mot (EXIT pour sortir): ");
		a = 0;
		while (1) {

			st1[a] = fgetc(stdin);

			if ((st1[a] == '\n') || (a >= max_size - 1)) {

				st1[a] = 0;

				break;
			}
			a++;
		}

		if (!strcmp(st1, "EXIT")) break;

		b = 0;
		c = 0;
		while (1) {
			st[b] = st1[c];
			b++;
			c++;
			st[b] = 0;
			if (st1[c] == 0) break;
		}
		iterateur = mots.find((char *)st);
		if (iterateur == mots.end())
			bi = -1;
		else
			bi = iterateur->second;

		printf("\nMot: %s  Position dans le vocabulaire: %lld\n", st, bi);
		if (bi == -1) {
			printf("Mot inconnu du dictionnaire!\n");
			continue;
		}

		for (a = 0; a < MAX; a++) bestd[a] = -1;
		for (a = 0; a < MAX; a++) bestw[a] = 0;
		
		for (c = 0; c < id_sens[bi]; c++) {

			for (a = 0; a < size; a++) vec_1[a] = M[a + size * ((bi * nombreDeSynonyme) + c)];
			len = 0;
			for (a = 0; a < size; a++)	len += vec_1[a] * vec_1[a];

			len = sqrt(len);
			for (a = 0; a < size; a++) vec_1[a] /= len;

			for (b = 0; b < words; b++)
			{
				if (b == bi)
					continue;

				for (d = 0; d < id_sens[b]; d++) {
					dist = 0;
					for (a = 0; a < size; a++) 
						dist += vec_1[a] * M[a + size * ((b * nombreDeSynonyme) + d)];
					
					prob = probas[(bi * nombreDeSynonyme) + c] * probas[(b * nombreDeSynonyme) + d];

					for (a = 0; a < MAX; a++) {
						if (dist > bestd[a]) {

							for (e = MAX - 1; e > a; e--) {
								bestd[e] = bestd[e - 1];
								probas_sel[e] = probas_sel[e - 1];
								bestw[e] = bestw[e - 1];
							}
							bestd[a] = dist;
							
							probas_sel[a] = prob;
							bestw[a] = b;
							break;
						}
					}
				}
			}
		}

		for (a = 0; a < MAX; a++)
				printf("Mot = %-20s Distance = %.6f	Probabilite = %f\n",id_mots[bestw[a]] ,bestd[a], probas_sel[a]);
		printf("\n");
	}
	return 0;
}
