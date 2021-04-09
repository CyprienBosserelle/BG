//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//Copyright (C) 2018 Bosserelle                                                 //
//                                                                              //
//This program is free software: you can redistribute it and/or modify          //
//it under the terms of the GNU General Public License as published by          //
//the Free Software Foundation.                                                 //
//                                                                              //
//This program is distributed in the hope that it will be useful,               //
//but WITHOUT ANY WARRANTY; without even the implied warranty of                //    
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                 //
//GNU General Public License for more details.                                  //
//                                                                              //
//You should have received a copy of the GNU General Public License             //
//along with this program.  If not, see <http://www.gnu.org/licenses/>.         //
//////////////////////////////////////////////////////////////////////////////////

#include "ReadInput.h"

// Collection of functions to read input to the model


/*! \fn T readfileinfo(std::string input,T outinfo)
* convert file name into name and extension
* This is used for various input classes
*
* template inputmap readfileinfo<inputmap>(std::string input, inputmap outinfo);
* template forcingmap readfileinfo<forcingmap>(std::string input, forcingmap outinfo);
* template StaticForcingP<float> readfileinfo<StaticForcingP<float>>(std::string input, StaticForcingP<float> outinfo);
* template DynForcingP<float> readfileinfo<DynForcingP<float>>(std::string input, DynForcingP<float> outinfo);
* template deformmap<float> readfileinfo<deformmap<float>>(std::string input, deformmap<float> outinfo);
*/
template <class T> T readfileinfo(std::string input,T outinfo)
{
	// Outinfo is based on an inputmap (or it's sub classes)
	
	//filename include the file extension

	std::vector<std::string> extvec = split(input, '.');

	//outinfo.inputfile = extvec.front();

	std::vector<std::string> nameelements;
	//
	nameelements = split(extvec.back(), '?');
	if (nameelements.size() > 1)
	{
		//variable name for bathy is not given so it is assumed to be zb
		outinfo.extension = nameelements[0];
		outinfo.varname = nameelements.back();

	}
	else
	{
		outinfo.extension = extvec.back();
	}

	//Reconstruct filename with extension but without varname
	outinfo.inputfile = extvec.front() + "." + outinfo.extension;


	return outinfo;
}

template inputmap readfileinfo<inputmap>(std::string input, inputmap outinfo);
template forcingmap readfileinfo<forcingmap>(std::string input, forcingmap outinfo);
template StaticForcingP<float> readfileinfo<StaticForcingP<float>>(std::string input, StaticForcingP<float> outinfo);
template DynForcingP<float> readfileinfo<DynForcingP<float>>(std::string input, DynForcingP<float> outinfo);
template deformmap<float> readfileinfo<deformmap<float>>(std::string input, deformmap<float> outinfo);




/*! \fn void Readparamfile(Param &XParam, Forcing<float> & XForcing)
* Open the BG_param.txt file and read the parameters
* save the parameter in the Param class and or Forcing class.
*/
void Readparamfile(Param &XParam, Forcing<float> & XForcing)
{
	//
	log("\nReading BG_param.txt ...");
	std::ifstream fs("BG_param.txt");

	if (fs.fail()) {
		//std::cerr << "BG_param.txt file could not be opened" << std::endl;
		log("ERROR: BG_param.txt file could not be opened...use this log file to create a file named BG_param.txt");
		SaveParamtolog(XParam);

		exit(1);

	}
	else
	{
		// Read and interpret each line of the BG_param.txt
		std::string line;
		while (std::getline(fs, line))
		{

			//Get param or skip empty lines
			if (!line.empty() && line.substr(0, 1).compare("#") != 0)
			{
				XParam = readparamstr(line, XParam);
				XForcing = readparamstr(line, XForcing);

				//std::cout << line << std::endl;
			}

		}
		fs.close();


	}
	
}





/*! \fn Param readparamstr(std::string line, Param param)
* Read BG_param.txt line and convert parameter to the righ parameter in teh class
* retrun an updated Param class 
*/
Param readparamstr(std::string line, Param param)
{


	std::string parameterstr, parametervalue;
	//std::vector<std::string> paramvec;
	///////////////////////////////////////////////////////
	// General parameters
	//


	parameterstr = "test";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.test = std::stoi(parametervalue);
	}

	std::vector<std::string> paramvec = { "gpudevice","GPUDEVICE" };
	parametervalue = findparameter(paramvec, line);
	if (!parametervalue.empty())
	{
		param.GPUDEVICE = std::stoi(parametervalue);
	}
	/*
	parameterstr = "GPUDEVICE";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.GPUDEVICE = std::stoi(parametervalue);
	}
	*/
	parameterstr = "doubleprecision";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.doubleprecision = std::stoi(parametervalue);
	}
	///////////////////////////////////////////////////////
	// Adaptation
	//
	parameterstr = "maxlevel";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.maxlevel = std::stoi(parametervalue);
	}

	parameterstr = "minlevel";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.minlevel = std::stoi(parametervalue);
	}

	parameterstr = "initlevel";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.initlevel = std::stoi(parametervalue);
	}

	parameterstr = "conserveelevation";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{

		if (parametervalue.compare("true") == 0 || parametervalue.compare("True") == 0)
		{
			param.conserveElevation = true;
		}
		else if (parametervalue.compare("false") == 0 || parametervalue.compare("False") == 0)
		{
			param.conserveElevation = false;
		}


	}


	parameterstr = "membuffer";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.membuffer = std::stod(parametervalue);
	}

	///////////////////////////////////////////////////////
	// Flow parameters
	//
	parameterstr = "eps";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.eps = std::stod(parametervalue);
	}
	
	parameterstr = "cf";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.cf = std::stod(parametervalue);
	}

	
	
	parameterstr = "Cd";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.Cd = std::stod(parametervalue);
	}

	parameterstr = "Pa2m";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.Pa2m = std::stod(parametervalue);
	}

	parameterstr = "Paref";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.Paref = std::stod(parametervalue);
	}

	parameterstr = "mask";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.mask = std::stod(parametervalue);
	}

	///////////////////////////////////////////////////////
	// Timekeeping parameters
	//
	parameterstr = "dt";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.dt = std::stod(parametervalue);

	}

	parameterstr = "CFL";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.CFL = std::stod(parametervalue);

	}
	parameterstr = "theta";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.theta = std::stod(parametervalue);

	}

	
	parameterstr = "outputtimestep";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.outputtimestep = std::stod(parametervalue);

	}
	parameterstr = "outtimestep";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.outputtimestep = std::stod(parametervalue);

	}

	parameterstr = "endtime";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.endtime = std::stod(parametervalue);

	}
	parameterstr = "totaltime";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.totaltime = std::stod(parametervalue);

	}

	///////////////////////////////////////////////////////
	// Input and output files
	//
	
	parameterstr = "outfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.outfile = parametervalue;
		
	}

	
	// Below is a bit more complex than usual because more than 1 node can be outputed as a timeseries
	parameterstr = "TSOutput";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		TSoutnode node;
		std::vector<std::string> nodeitems = split(parametervalue, ',');
		if (nodeitems.size() > 3)
		{
			node.outname = nodeitems[0];
			node.x = std::stod(nodeitems[1]);
			node.y = std::stod(nodeitems[2]);

			param.TSnodesout.push_back(node);
		}
		else
		{
			std::cerr << "Node input failed there should be 3 arguments (comma separated) when inputing a outout node: TSOutput = filename, xvalue, yvalue; see log file for details" << std::endl;

			log("Node input failed there should be 3 arguments (comma separated) when inputing a outout node: TSOutput = filename, xvalue, yvalue; see log file for details. Input was: "+ parametervalue);
			
		}
		
	}
	


	//outvars
	parameterstr = "outvars";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> vars = split(parametervalue, ',');
		for (int nv = 0; nv < vars.size(); nv++)
		{
			//Verify that the variable name makes sense?
			//Need to add more here
			std::vector<std::string> SupportedVarNames = { "zb", "zs", "u", "v", "h", "hmean", "zsmean", "umean", "vmean", "hmax", "zsmax", "umax", "vmax" ,"vort","dhdx","dhdy","dzsdx","dzsdy","dudx","dudy","dvdx","dvdy","Fhu","Fhv","Fqux","Fqvy","Fquy","Fqvx","Su","Sv","dh","dhu","dhv","cf"};
			std::string vvar = trim(vars[nv], " ");
			for (int isup = 0; isup < SupportedVarNames.size(); isup++)
			{
				
				//std::cout << "..." << vvar << "..." << std::endl;
				if (vvar.compare(SupportedVarNames[isup]) == 0)
				{
					param.outvars.push_back(vvar);
					break;
				}

			}

			param.outmean = (vvar.compare("hmean") == 0) ? true : param.outmean;
			param.outmean = (vvar.compare("zsmean") == 0) ? true : param.outmean;
			param.outmean = (vvar.compare("umean") == 0) ? true : param.outmean;
			param.outmean = (vvar.compare("vmean") == 0) ? true : param.outmean;

			param.outmax = (vvar.compare("hmax") == 0) ? true : param.outmax;
			param.outmax = (vvar.compare("zsmax") == 0) ? true : param.outmax;
			param.outmax = (vvar.compare("umax") == 0) ? true : param.outmax;
			param.outmax = (vvar.compare("vmax") == 0) ? true : param.outmax;

			param.outvort = (vvar.compare("vort") == 0) ? true : param.outvort;
		}
		

		
	}

	


	parameterstr = "resetmax";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.resetmax = std::stoi(parametervalue);
	}


	

	parameterstr = "nx";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.nx = std::stoi(parametervalue);
	}

	parameterstr = "ny";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.ny = std::stoi(parametervalue);
	}

	parameterstr = "dx";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.dx = std::stod(parametervalue);
	}

	parameterstr = "grdalpha";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.grdalpha = std::stod(parametervalue);
	}

	parameterstr = "xo";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.xo = std::stod(parametervalue);
	}
	parameterstr = "xmin";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.xo = std::stod(parametervalue);
	}

	parameterstr = "yo";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.yo = std::stod(parametervalue);
	}
	parameterstr = "ymin";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.yo = std::stod(parametervalue);
	}

	parameterstr = "xmax";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.xmax = std::stod(parametervalue);
	}

	parameterstr = "ymax";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.ymax = std::stod(parametervalue);
	}

	parameterstr = "g";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.g = std::stod(parametervalue);
		
	}

	parameterstr = "rho";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.rho = std::stod(parametervalue);
	}

	parameterstr = "smallnc";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.smallnc = std::stoi(parametervalue);
	}
	parameterstr = "scalefactor";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.scalefactor = std::stof(parametervalue);
	}
	parameterstr = "addoffset";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.addoffset = std::stof(parametervalue);
	}
	parameterstr = "posdown";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.posdown = std::stoi(parametervalue);
	}

#ifdef USE_CATALYST
	parameterstr = "use_catalyst";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.use_catalyst = std::stoi(parametervalue);
	}
	parameterstr = "catalyst_python_pipeline";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.catalyst_python_pipeline = std::stoi(parametervalue);
	}
	parameterstr = "vtk_output_frequency";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.vtk_output_frequency = std::stoi(parametervalue);
	}
	parameterstr = "vtk_output_time_interval";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.vtk_output_time_interval = std::stod(parametervalue);
	}
	parameterstr = "vtk_outputfile_root";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.vtk_outputfile_root = parametervalue;
	}
	parameterstr = "python_pipeline";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.python_pipeline = parametervalue;
	}
#endif

	parameterstr = "initzs";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.zsinit = std::stod(parametervalue);
	}

	parameterstr = "zsinit";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.zsinit = std::stod(parametervalue);
	}

	parameterstr = "zsoffset";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.zsoffset = std::stod(parametervalue);
	}

	parameterstr = "hotstartfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.hotstartfile = parametervalue;
		
	}
	
	parameterstr = "hotstep";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.hotstep = std::stoi(parametervalue);
	}
	

	parameterstr = "spherical";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.spherical = std::stoi(parametervalue);
	}

	parameterstr = "Radius";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.Radius = std::stod(parametervalue);
	}

	parameterstr = "frictionmodel";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		param.frictionmodel = std::stoi(parametervalue);
	}
	
	parameterstr = "Adaptation";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> adaptpar = split(parametervalue, ',');

		if (!adaptpar.empty())
		{
			param.AdatpCrit = adaptpar[0];
			if (adaptpar.size() > 1)
				param.Adapt_arg1 = adaptpar[1];
			if (adaptpar.size() > 2)
				param.Adapt_arg2 = adaptpar[2];
			if (adaptpar.size() > 3)
				param.Adapt_arg3 = adaptpar[3];
			if (adaptpar.size() > 4)
				param.Adapt_arg4 = adaptpar[4];
			if (adaptpar.size() > 5)
				param.Adapt_arg5 = adaptpar[5];
		}
	}

	return param;
}



/*! \fn Forcing<T> readparamstr(std::string line, Forcing<T> forcing)
* Read BG_param.txt line and convert parameter to the righ parameter in the class
* return an updated Param class
*/
template <class T>
Forcing<T> readparamstr(std::string line, Forcing<T> forcing)
{
	std::string parameterstr, parametervalue;

	parameterstr = "bathy";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		StaticForcingP<float> infobathy;
		forcing.Bathy.push_back(readfileinfo(parametervalue, infobathy));
		//std::cerr << "Bathymetry file found!" << std::endl;
	}

	parameterstr = "bathyfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		StaticForcingP<float> infobathy;
		forcing.Bathy.push_back(readfileinfo(parametervalue, infobathy));
		//forcing.Bathy = readfileinfo(parametervalue, forcing.Bathy);
		//std::cerr << "Bathymetry file found!" << std::endl;
	}

	parameterstr = "bathymetry";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		StaticForcingP<float> infobathy;
		forcing.Bathy.push_back(readfileinfo(parametervalue, infobathy));
		//forcing.Bathy = readfileinfo(parametervalue, forcing.Bathy);
		//std::cerr << "Bathymetry file found!" << std::endl;
	}

	//
	parameterstr = "depfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		StaticForcingP<float> infobathy;
		forcing.Bathy.push_back(readfileinfo(parametervalue, infobathy));
		//forcing.Bathy = readfileinfo(parametervalue, forcing.Bathy);
	}


	// Boundaries
	parameterstr = "leftbndfile";// or left or leftbnd
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> items = split(parametervalue, ',');
		if (items.size() == 1)
		{
			forcing.left.type = std::stoi(items[0]);
			
		}
		else if (items.size() >= 2)
		{
			forcing.left.type = std::stoi(items[1]);
			forcing.left.inputfile = items[0];
			forcing.left.on = true;
		}
				
	}

	parameterstr = "leftbnd";// or left or leftbnd
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> items = split(parametervalue, ',');
		if (items.size() == 1)
		{
			forcing.left.type = std::stoi(items[0]);

		}
		else if (items.size() >= 2)
		{
			forcing.left.type = std::stoi(items[1]);
			forcing.left.inputfile = items[0];
			forcing.left.on = true;
		}

	}
	parameterstr = "left";// or left or leftbnd
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> items = split(parametervalue, ',');
		if (items.size() == 1)
		{
			forcing.left.type = std::stoi(items[0]);

		}
		else if (items.size() >= 2)
		{
			forcing.left.type = std::stoi(items[1]);
			forcing.left.inputfile = items[0];
			forcing.left.on = true;
		}

	}

	
	parameterstr = "rightbndfile";// or left or leftbnd
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> items = split(parametervalue, ',');
		if (items.size() == 1)
		{
			forcing.right.type = std::stoi(items[0]);

		}
		else if (items.size() >= 2)
		{
			forcing.right.type = std::stoi(items[1]);
			forcing.right.inputfile = items[0];
			forcing.right.on = true;
		}

	}
	parameterstr = "rightbnd";// or left or leftbnd
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> items = split(parametervalue, ',');
		if (items.size() == 1)
		{
			forcing.right.type = std::stoi(items[0]);

		}
		else if (items.size() >= 2)
		{
			forcing.right.type = std::stoi(items[1]);
			forcing.right.inputfile = items[0];
			forcing.right.on = true;
		}

	}
	parameterstr = "right";// or left or leftbnd
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> items = split(parametervalue, ',');
		if (items.size() == 1)
		{
			forcing.right.type = std::stoi(items[0]);

		}
		else if (items.size() >= 2)
		{
			forcing.right.type = std::stoi(items[1]);
			forcing.right.inputfile = items[0];
			forcing.right.on = true;
		}

	}

	parameterstr = "topbndfile";// or left or leftbnd
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> items = split(parametervalue, ',');
		if (items.size() == 1)
		{
			forcing.top.type = std::stoi(items[0]);

		}
		else if (items.size() >= 2)
		{
			forcing.top.type = std::stoi(items[1]);
			forcing.top.inputfile = items[0];
			forcing.top.on = true;
		}

	}
	parameterstr = "top";// or left or leftbnd
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> items = split(parametervalue, ',');
		if (items.size() == 1)
		{
			forcing.top.type = std::stoi(items[0]);

		}
		else if (items.size() >= 2)
		{
			forcing.top.type = std::stoi(items[1]);
			forcing.top.inputfile = items[0];
			forcing.top.on = true;
		}

	}
	parameterstr = "topbnd";// or left or leftbnd
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> items = split(parametervalue, ',');
		if (items.size() == 1)
		{
			forcing.top.type = std::stoi(items[0]);

		}
		else if (items.size() >= 2)
		{
			forcing.top.type = std::stoi(items[1]);
			forcing.top.inputfile = items[0];
			forcing.top.on = true;
		}

	}

	parameterstr = "botbndfile";// or left or leftbnd
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> items = split(parametervalue, ',');
		if (items.size() == 1)
		{
			forcing.bot.type = std::stoi(items[0]);

		}
		else if (items.size() >= 2)
		{
			forcing.bot.type = std::stoi(items[1]);
			forcing.bot.inputfile = items[0];
			forcing.bot.on = true;
		}

	}
	parameterstr = "botbnd";// or left or leftbnd
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> items = split(parametervalue, ',');
		if (items.size() == 1)
		{
			forcing.bot.type = std::stoi(items[0]);

		}
		else if (items.size() >= 2)
		{
			forcing.bot.type = std::stoi(items[1]);
			forcing.bot.inputfile = items[0];
			forcing.bot.on = true;
		}

	}
	parameterstr = "bot";// or left or leftbnd
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> items = split(parametervalue, ',');
		if (items.size() == 1)
		{
			forcing.bot.type = std::stoi(items[0]);

		}
		else if (items.size() >= 2)
		{
			forcing.bot.type = std::stoi(items[1]);
			forcing.bot.inputfile = items[0];
			forcing.bot.on = true;
		}

	}


	//Tsunami deformation input files
	parameterstr = "deform";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{

		deformmap<float> thisdeform;
		std::vector<std::string> items = split(parametervalue, ',');
		//Need sanity check here
		thisdeform = readfileinfo(items[0], thisdeform);
		//thisdeform.inputfile = items[0];
		if (items.size() > 1)
		{
			thisdeform.startime = std::stod(items[1]);

		}
		if (items.size() > 2)
		{
			thisdeform.duration = std::stod(items[2]);

		}

		forcing.deform.push_back(thisdeform);

	}

	//River
	parameterstr = "river";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> vars = split(parametervalue, ',');
		if (vars.size() == 5)
		{
			River thisriver;
			thisriver.Riverflowfile = trim(vars[0], " ");
			thisriver.xstart = std::stod(vars[1]);
			thisriver.xend = std::stod(vars[2]);
			thisriver.ystart = std::stod(vars[3]);
			thisriver.yend = std::stod(vars[4]);

			forcing.rivers.push_back(thisriver);
		}
		else
		{
			//Failed there should be 5 arguments (comma separated) when inputing a river: filename, xstart,xend,ystart,yend;
			std::cerr << "River input failed there should be 5 arguments (comma separated) when inputing a river: river = filename, xstart,xend,ystart,yend; see log file for details" << std::endl;

			log("River input below failed there should be 5 arguments (comma separated) when inputing a river: river = filename, xstart,xend,ystart,yend;");
			log(parametervalue);
		}
	}

	// Mapped friction
	parameterstr = "cfmap";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{

		forcing.cf = readfileinfo(parametervalue, forcing.cf);

	}
	parameterstr = "roughnessmap";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{

		forcing.cf = readfileinfo(parametervalue, forcing.cf);

	}

	// wind forcing
	parameterstr = "windfiles";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{

		std::vector<std::string> vars = split(parametervalue, ',');
		if (vars.size() == 2)
		{
			// If 2 parameters (files) are given then 1st file is U wind and second is V wind.
			// This is for variable winds no rotation of the data is performed
			
			forcing.UWind = readfileinfo(trim(vars[0], " "), forcing.UWind);
			forcing.VWind = readfileinfo(trim(vars[1], " "), forcing.VWind);
		}
		else if (vars.size() == 1)
		{
			// if 1 parameter(file) is given then a 3 column file is expected showing time windspeed and direction
			// wind direction is rotated (later) to the grid direction (via grdalfa)
			forcing.UWind = readfileinfo(parametervalue, forcing.UWind);
			forcing.UWind.uniform = 1;
			
			//apply the same for Vwind? seem unecessary but need to be careful later in the code
		}
		else
		{
			//Failed there should be 5 arguments (comma separated) when inputing a river: filename, xstart,xend,ystart,yend;
			//std::cerr << "Wind input failed there should be 2 arguments (comma separated) when inputing a wind: windfiles = windfile.nc?uwind, windfile.nc?vwind; see log file for details" << std::endl;

			log("Wind input failed there should be 2 arguments(comma separated) when inputing a wind : windfiles = windfile.nc ? uwind, windfile.nc ? vwind; see log file for details");
			log(parametervalue);
		}

	}

	// atmpress forcing
	parameterstr = "atmpfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		// needs to be a netcdf file 
		forcing.Atmp = readfileinfo(parametervalue, forcing.Atmp);
	}

	// Rain forcing
	parameterstr = "rainfile";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		// netcdf file == Variable spatially
		// txt file (other than .nc) == spatially cst (txt file with 2 col time and mmm/h )
		forcing.Rain = readfileinfo(parametervalue, forcing.Rain);
		
		//set the expected type of input

		if (forcing.Rain.extension.compare("nc") == 0)
		{
			forcing.Rain.uniform = 0;
		}
		else
		{
			forcing.Rain.uniform = 1;
		}



	}

	parameterstr = "Adaptation";
	parametervalue = findparameter(parameterstr, line);
	if (!parametervalue.empty())
	{
		std::vector<std::string> adaptpar = split(parametervalue, ',');
		// special case for 'Targetlevel' adaptation
		if (!adaptpar.empty())
		{
			if (adaptpar[0].compare("Targetlevel") == 0)
			{
				for (int ng = 1; ng < adaptpar.size(); ng++)
				{
					StaticForcingP<int> infogrid;
					forcing.targetadapt.push_back(readfileinfo(adaptpar[ng], infogrid));
				}
			}
		}

	}

	return forcing;
}


/*! \fn void checkparamsanity(Param & XParam, Forcing<float> & XForcing)
* Check the Sanity of both Param and Forcing class
* If required some parameter are infered 
*/
void checkparamsanity(Param & XParam, Forcing<float> & XForcing)
{
	Param DefaultParams;

	double tiny = 0.0000001;

	// Sanity check for model levels
	int minlev = XParam.minlevel;
	int maxlev = XParam.maxlevel;

	XParam.maxlevel = utils::max(maxlev, minlev);
	XParam.minlevel = utils::min(maxlev, minlev);

	XParam.initlevel = utils::min(utils::max(XParam.minlevel, XParam.initlevel), XParam.maxlevel);

	//force double for Rain on grid cases
	if (!XForcing.Rain.inputfile.empty())
	{
		XParam.doubleprecision = 1;
	}

	XParam.blkmemwidth = XParam.blkwidth + 2 * XParam.halowidth;
	XParam.blksize = utils::sq(XParam.blkmemwidth);

	///////////////////////////////////////////
	//  Read Bathy Information
	///////////////////////////////////////////

	//this sets xo yo  etc...

	// Any of xo,yo,xmax,ymax or dx not defined is assigned the value from bathy file
	//default value is nan in default param file

	//inputmap Bathymetry;
	//Bathymetry.inputfile = XForcing.Bathy.inputfile;
	//XForcing.Bathy = readforcinghead(XForcing.Bathy);
	


	if (std::isnan(XParam.xo))
		XParam.xo = XForcing.Bathy[0].xo-(0.5* XForcing.Bathy[0].dx);
	if (std::isnan(XParam.xmax))
		XParam.xmax = XForcing.Bathy[0].xmax + (0.5 * XForcing.Bathy[0].dx);
	if(std::isnan(XParam.yo))
		XParam.yo = XForcing.Bathy[0].yo - (0.5 * XForcing.Bathy[0].dx);
	if (std::isnan(XParam.ymax))
		XParam.ymax = XForcing.Bathy[0].ymax + (0.5 * XForcing.Bathy[0].dx);

	if (std::isnan(XParam.dx))
		XParam.dx = XForcing.Bathy[0].dx;
	
	if (std::isnan(XParam.grdalpha))
		XParam.grdalpha = XForcing.Bathy[0].grdalpha; // here the default bathy grdalpha is 0.0 as defined by inputmap/Bathymetry class


	//Check Bathy input type
	if (XForcing.Bathy[0].extension.compare("dep") == 0 || XForcing.Bathy[0].extension.compare("bot") == 0)
	{
		if (std::isnan(XParam.dx))
		{
			//std::cerr << "FATAL ERROR: nx or ny or dx were not specified. These parameters are required when using ." << bathyext << " file" << std::endl;
			log("FATAL ERROR: nx or ny or dx were not specified. These parameters are required when using ." + XForcing.Bathy[0].extension + " file");
			exit(1);
		}
	}

	double levdx = calcres(XParam.dx, XParam.initlevel);// true grid resolution as in dx/2^(initlevel)
	//printf("levdx=%f;1 << XParam.initlevel=%f\n", levdx, calcres(1.0, XParam.initlevel));

	// First estimate nx and ny
	XParam.nx = (XParam.xmax - XParam.xo) / (levdx);
	XParam.ny = (XParam.ymax - XParam.yo) / (levdx); //+1?


	// Adjust xmax and ymax so that nx and ny are a factor of XParam.blkwidth [16]
	XParam.xmax = XParam.xo + (ceil(XParam.nx / ((double)XParam.blkwidth)) * ((double)XParam.blkwidth)) * levdx;
	XParam.ymax = XParam.yo + (ceil(XParam.ny / ((double)XParam.blkwidth)) * ((double)XParam.blkwidth)) * levdx;

	// Update nx and ny 
	XParam.nx = (XParam.xmax - XParam.xo) / (levdx);
	XParam.ny = (XParam.ymax - XParam.yo) / (levdx); //+1?

	log("\nAdjusted model domain (xo/xmax/yo/ymax): ");
	log("\t" + std::to_string(XParam.xo) + "/" + std::to_string(XParam.xmax) + "/" + std::to_string(XParam.yo) + "/" + std::to_string(XParam.ymax) );
	log("\t Initial resolution (level " + std::to_string(XParam.initlevel) + ") = " + std::to_string(levdx) );

	if (XParam.spherical < 1)
	{
		XParam.delta = XParam.dx;
		XParam.grdalpha = XParam.grdalpha*pi / 180.0; // grid rotation

	}
	else
	{
		//Geo grid
		XParam.delta = XParam.dx * XParam.Radius*pi / 180.0;
		//printf("Using spherical coordinate; delta=%f rad\n", XParam.delta);
		log("Using spherical coordinate; delta=" + std::to_string(XParam.delta));
		if (XParam.grdalpha != 0.0)
		{
			//printf("grid rotation in spherical coordinate is not supported yet. grdalpha=%f rad\n", XParam.grdalpha);
			log("grid rotation in spherical coordinate is not supported yet. grdalpha=" + std::to_string(XParam.grdalpha*180.0 / pi));
		}
	}

	//setup extra infor about boundaries
	XForcing.left.side = 3;
	XForcing.left.isright = -1;
	XForcing.left.istop = 0;

	XForcing.right.side = 1;
	XForcing.right.isright = 1;
	XForcing.right.istop = 0;

	XForcing.top.side = 0;
	XForcing.top.isright = 0;
	XForcing.top.istop = 1;

	XForcing.bot.side = 2;
	XForcing.bot.isright = 0;
	XForcing.bot.istop = -1;


	// Make sure the nriver in param (used for preallocation of memory) and number of rivers in XForcing are consistent
	XParam.nrivers = XForcing.rivers.size();



	// Check whether endtime was specified by the user
	//No; i.e. endtimne =0.0
	//so the following conditions are useless
	
	
	
	if (abs(XParam.endtime - DefaultParams.endtime) <= tiny)
	{
		//No; i.e. endtimne =0.0
		XParam.endtime = 1.0 / tiny; //==huge
	}

	XParam.endtime = setendtime(XParam, XForcing);
	
		
	
	
	




	// Check that outputtimestep is not zero, so at least the first and final time step are saved
	// If only the model stepup is needed than just run with endtime=0.0
	if (abs(XParam.outputtimestep - DefaultParams.outputtimestep) <= tiny)
	{
		XParam.outputtimestep = XParam.endtime;
		//otherwise there is really no point running the model
	}

	

	if (XParam.outvars.empty() && XParam.outputtimestep > 0.0)
	{
		//a nc file was specified but no output variable were specified
		std::vector<std::string> SupportedVarNames = { "zb", "zs", "u", "v", "h" }; 
		for (int isup = 0; isup < SupportedVarNames.size(); isup++)
		{
			XParam.outvars.push_back(SupportedVarNames[isup]);
				
		}

	}


	// CHeck whether a cuda compatible GPU is present
	if (XParam.GPUDEVICE >= 0)
	{
		// Init GPU
		// This should be in the sanity check
		int nDevices;
		cudaGetDeviceCount(&nDevices);
		cudaDeviceProp prop;

		if (XParam.GPUDEVICE > (nDevices - 1))
		{
			//  if no GPU device are present then use the CPU (GPUDEVICE = -1)
			XParam.GPUDEVICE = (nDevices - 1);
		}
		cudaGetDeviceProperties(&prop, XParam.GPUDEVICE);
		//printf("There are %d GPU devices on this machine\n", nDevices);
		log("There are " + std::to_string(nDevices) + " GPU devices on this machine");

		if (XParam.GPUDEVICE >= 0)
		{
			
			log("Using Device: " + std::string(prop.name));
		}
		else
		{
			log("No GPU device were detected on this machine... Using CPU instead");
		}

	}


	if (XParam.minlevel != XParam.maxlevel)
	{
		if (XParam.AdatpCrit.empty())
		{
			XParam.AdatpCrit = "Threshold";
			XParam.Adapt_arg1 = "0.0";
			XParam.Adapt_arg2 = "h";
		}
	}
	
}

/*! \fn double setendtime(Param XParam,Forcing<float> XForcing)
* Calculate/modify endtime based on maximum time in forcing 
*
*/
double setendtime(Param XParam,Forcing<float> XForcing)
{
	//endtime cannot be bigger thn the smallest time set in a boundary
	SLTS tempSLTS;
	double endtime = XParam.endtime;
	if (XForcing.left.on)
	{
		tempSLTS =XForcing.left.data.back();
		endtime = utils::min( endtime, tempSLTS.time);
		
	}
	if (XForcing.right.on)
	{
		tempSLTS = XForcing.right.data.back();
		endtime = utils::min(endtime, tempSLTS.time);
	}
	if (XForcing.top.on)
	{
		tempSLTS = XForcing.top.data.back();
		endtime = utils::min(endtime, tempSLTS.time);
	}
	if (XForcing.bot.on)
	{
		tempSLTS = XForcing.bot.data.back();
		endtime = utils::min(endtime, tempSLTS.time);
	}

	return endtime;
}

/*! \fn std::string findparameter(std::string parameterstr, std::string line)
* separate parameter from value
*
*/
std::string findparameter(std::vector<std::string> parameterstr, std::string line)
{
	std::size_t found;
	std::string parameternumber,left,right;
	std::vector<std::string> splittedstr;
	
	// first look fo an equal sign
	// No equal sign mean not a valid line so skip
	splittedstr=split(line, '=' );
	if (splittedstr.size()>1)
	{
		left = trim(splittedstr[0]," ");
		right = splittedstr[1]; // if there are more than one equal sign in the line the second one is ignored
		for (int ii = 0; ii < parameterstr.size(); ii++)
		{
			found = left.compare(parameterstr[ii]);// it needs to strictly compare
			if (found == 0)
				break;
		}
		if (found == 0) // found the parameter
		{
			//std::cout <<"found LonMin at : "<< found << std::endl;
			//Numberstart = found + parameterstr.length();
			splittedstr = split(right, ';');
			if (splittedstr.size() >= 1)
			{
				parameternumber = splittedstr[0];
			}
			//std::cout << parameternumber << std::endl;

		}
	}
	return trim(parameternumber, " ");
}


std::string findparameter(std::string parameterstr, std::string line)
{
	std::vector<std::string> parametervec;

	parametervec.push_back(parameterstr);
	return findparameter(parametervec, line);
}


/*! \fn void split(const std::string &s, char delim, std::vector<std::string> &elems)
* split string based in character
*
*/
void split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		if (!item.empty())//skip empty tokens
		{
			elems.push_back(item);
		}
		
	}
}

/*! \fn std::vector<std::string> split(const std::string &s, char delim)
* split string based in character
*
*/
std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}


/*! \fn std::string trim(const std::string& str, const std::string& whitespace)
* remove leading and trailing space in a string
*
*/
std::string trim(const std::string& str, const std::string& whitespace)
{
	const auto strBegin = str.find_first_not_of(whitespace);
	if (strBegin == std::string::npos)
		return ""; // no content

	const auto strEnd = str.find_last_not_of(whitespace);
	const auto strRange = strEnd - strBegin + 1;

	return str.substr(strBegin, strRange);
}




