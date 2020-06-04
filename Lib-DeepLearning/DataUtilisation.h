#pragma once
#include<fstream>
#include<vector>
#include<functional>
#include<algorithm>
#include<string>



class DataUtilisation
{
	template<typename T>
	static bool convertString(T &, std::string);
	
public:

	template<typename T>
	static short CSVReader(std::ifstream, char, std::vector<std::vector<T>> &, bool = false);
};

template<typename T>
inline bool DataUtilisation::convertString(T& variable, std::string str)
{
	std::stringstream stream(str);
	str >> variable;

	return stream.fail();
}

template<typename T>

/*
Return error types :
-0 -> no problem
-1-> file access error;
-2-> empty file (or nothing has been extract);
*/

inline short DataUtilisation::CSVReader(std::ifstream file, char delimit, std::vector<std::vector<T>> &CSVList, bool deleteLigne)
{
	CSVList.clear();
	CSVList.push_back(std::vector<T>(1));
	
	char cBuffer = '\0';
	std::string stBuffer("");
	bool inLigneDeltetion = false;
	bool firstCharAlready = false;

	auto MainIt = CSVList.begin();
	auto SecondIt = MainIt->begin();

	if (!file)
		return 1;

	do
	{
		file >> cBuffer;

		if (inLigneDeltetion && cBuffer != '\n')
			continue;

		switch (cBuffer)
		{			
			case '\n':
				if ((convertString(*SecondIt, stBuffer) && deleteLigne) || (inLigneDeltetion))
				{
					MainIt->clear();
					MainIt->push_back(0);
				}
				else
				{
					CSVList.push_back(std::vector<T>(1));
					MainIt++;
				}
				SecondIt = MainIt->begin();
				inLigneDeltetion = false;
				break;

			case EOF:
				if ((convertString(*SecondIt, stBuffer) && deleteLigne) || (inLigneDeltetion))
				{
					MainIt->clear();
					CSVList.pop_back();
				}

			case '\t':
			case ' ':
				if(!firstCharAlready)
					break;
			default:
				if (cBuffer == delimit)
				{
					if (convertString(*SecondIt, stBuffer) && deleteLigne)
						inLigneDeltetion = true;

					MainIt->push_back(0);
					SecondIt++;
					break;
				}

				firstCharAlready = true;
				stBuffer.append(cBuffer);
				break;
		}
	} while (cBuffer != EOF);

	if (CSVList.empty)
		return 2;
	else if (CSVList.size() == 1) if (CSVList[0].size() == 0 || CSVList[0].size() == 1)
		return 2;

	

	return 0;
}

template<>
short DataUtilisation::CSVReader<std::string>(std::ifstream file, char delimit, std::vector<std::vector<std::string>>& CSVList, bool deleteLigne)
{
	CSVList.clear();
	CSVList.push_back(std::vector<std::string>(1));

	char cBuffer = '\0';

	auto MainIt = CSVList.begin();
	auto SecondIt = MainIt->begin();

	if (!file)
		return 1;

	do
	{
		file >> cBuffer;

		switch (cBuffer)
		{
		case '\n':
			CSVList.push_back(std::vector<std::string>(1));
			MainIt++;
			SecondIt = MainIt->begin();
			break;

		case EOF:
			break;

		default:
			if (cBuffer == delimit)
			{
				MainIt->push_back("");
				SecondIt++;
				break;
			}
			SecondIt->append({ cBuffer });
			break;
		}
	} while (cBuffer != EOF);

	if (CSVList.empty())
		return 2;
	else if (CSVList.size() == 1) if (CSVList[0].size() == 0 || CSVList[0].size() == 1)
		return 2;

	return 0;
}
