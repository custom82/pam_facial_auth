#ifndef PAM_FACIAL_AUTH_UTILS_H
#define PAM_FACIAL_AUTH_UTILS_H

#include <dirent.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

class Utils
{
public:
	// Funzione per leggere la configurazione da un file
	static void GetConfig(const std::string &pathConfig, std::map<std::string, std::string> &config)
	{
		// Tenta di aprire il file di configurazione
		std::ifstream inFile(pathConfig);
		if (!inFile)
		{
			std::cerr << "Cannot open config file: " << pathConfig << std::endl;
			return;
		}

		std::string line;
		while (std::getline(inFile, line))
		{
			std::istringstream ssLine(line);
			std::string key;
			if (std::getline(ssLine, key, '='))
			{
				std::string value;
				if (std::getline(ssLine, value))
				{
					config[key] = value;
				}
			}
		}

		inFile.close();  // Assicurati che il file venga chiuso dopo la lettura
	}

	// Funzione per camminare nella directory e raccogliere file e sottodirectory
	static void WalkDirectory(const std::string &pathDir, std::vector<std::string> &files, std::vector<std::string> &subs)
	{
		DIR *dirMain;
		struct dirent *curr;

		// Apre la directory principale
		dirMain = opendir(pathDir.c_str());
		if (dirMain == nullptr)
		{
			std::cerr << "Cannot open directory: " << pathDir << std::endl;
			return;
		}

		// Itera attraverso gli elementi della directory
		while ((curr = readdir(dirMain)) != nullptr)
		{
			// Ignora i file "." e ".."
			if (std::strcmp(curr->d_name, ".") == 0 || std::strcmp(curr->d_name, "..") == 0)
			{
				continue;
			}

			// Costruisce il percorso completo
			std::string fullPath = pathDir + "/" + curr->d_name;

			// Controlla se è una directory o un file
			DIR *dirSub = opendir(fullPath.c_str());
			if (dirSub)  // Se è una directory
			{
				subs.push_back(curr->d_name);
				closedir(dirSub);
			}
			else  // Se è un file
			{
				files.push_back(curr->d_name);
			}
		}

		closedir(dirMain);  // Chiude la directory principale
	}
};

#endif //PAM_FACIAL_AUTH_UTILS_H
