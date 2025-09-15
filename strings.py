#!/usr/bin/env python3
#I made some progress on the script this weekend. 
# So far it will loop through a list of languages and then loop through all 
# screens/variations defined in an input file 
# (this will be the one created by you parsing the code). 
# It will then search the "zStringsVsScreens.txt" file and capture all of the strings 
# that are on the corresponding screen/var. 

#So you will still need to strip those strings out of the variable that is holding them, 
# and then use that to parse the language.xml file. 
# I can probably help with that part too if I have some time later this week. 
# Let me know if you want to walk through it at all
import subprocess
import sys
import os
import csv

Languages_To_Test = (
        ('Croation', 'LT'),
        ('Czech', 'LT'),
        ('Danish', 'LT'),
        ('English', 'LT')
    )

def run_executable(exe_path, arg1, arg2, arg3):
    """
    Run an executable file with three arguments.
    
    Args:
        exe_path (str): Path to the executable file
        arg1 (str): First argument
        arg2 (str): Second argument
        arg3 (str): Third argument
    
    Returns:
        subprocess.CompletedProcess: Result of the execution
    """
    
    # Check if executable exists
    if not os.path.exists(exe_path):
        print(f"Error: Executable not found at {exe_path}")
        return None
    
    # Check if file is executable (Unix-like systems)
    if not os.access(exe_path, os.X_OK):
        print(f"Warning: {exe_path} may not be executable")
    
    try:
        # Run the executable with arguments
        result = subprocess.run(
            [exe_path, arg1, arg2, arg3],
            capture_output=True,
            text=True,
            timeout=60,
            check=False  # Don't raise exception on non-zero exit code
        )
        
        ############## I think this is where we want to look if the exe opened correctly i.e. the screen/variation actually exists
        print(f"Command executed: {exe_path} {arg1} {arg2} {arg3}")
        print(f"Exit code: {result.returncode}")
        ############## Exit code 0 means it worked, not sure about failed cases
        
        # Print stdout if any
        if result.stdout:
            print("Output:")
            print(result.stdout)
        
        # Print stderr if any
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result
        
    except FileNotFoundError:
        print(f"Error: Could not find executable: {exe_path}")
        return None
    except Exception as e:
        print(f"Error running executable: {e}")
        return None
        
def GetListofScreensAndVariations():
    screens = []
    with open('StringsAndVariations.csv', newline='') as f:
        reader = csv.reader(f)
        next(reader) # skip the header, could do a check to make sure file is in expected format
        for row in reader:
            try:
                screens.append(row)
            except:
                print("Failed to Read Row in .csv")
                
    return screens
    
    
def LoopThroughScreens(executable_path, language, font):
    listOfScreens = GetListofScreensAndVariations()
    print(listOfScreens)
    
    for var in listOfScreens: # Start looping through all screens/variations
        screen = var[0]
        variation = var[1]
        
        # Start populating the cmd line string
        string = r'-screenNum=' # Create the last argument string
        string = string + str(screen) + ';' + str(variation) # make it look like -screenNum=screen;variation ex. -screenNum=700;1
        screen_var = string                # Third argument
        
        # Get the strings for the specific screen/variation we are looking at
        # Also note that this just returns a string right now, it will probably be easier to pick off the SID and put it into a tuple to pass to your screen compare tool
        stringsOnScreen = GetStringsForScreen_Var(screen, variation) # Need to do something with these in the screen check portion
        print(stringsOnScreen)
        
        # Run the executable
        # might have to figure out something slightly different here since it doesn't return until you close the window. 
        # Need to figure out how to tell if the exe successfully launched and then after we are done doing the image analysis, we must close the window and return
        result = run_executable(executable_path, language, font, screen_var)
        if result and result.returncode == 0:
            print("Program executed successfully!")
        elif result:
            print(f"Program finished with exit code {result.returncode}")
        else:
            print("Failed to execute program")
            
def GetStringsForScreen_Var(screen, var):
    foundVar = 0
    stringArray = []
    # Open the file in read mode
    with open('zStringsVsScreens.txt', 'r') as file:
        # Read all lines from the file
        lines = file.readlines()

    # Define the strings to search for
    string1 = "Screen = " + str(screen) + ","
    string2 = " Variation = " + str(var) + " uses the following SIDs:"

    # Iterate through each line
    for line in lines:
        # Check if we have the screen and variation we are looking for
        if string1 + string2 in line:
            #print(f"Found: '{string1} {string2}' in line: {line.strip()}")
            foundVar = 1
            continue # Go to the next line to start getting the lines with strings
        # Read until we hit a blank line, then we are done with our strings
        if line.strip() == '' and foundVar: 
            break
        # We found our screen and variation, now get the strings for it, read until we hit a blank line
        if foundVar:   
            stringArray.append(line.strip())

    return stringArray
    

def LoopThroughLanguages():
    # Exe name
    executable_path = "DesktopCP.exe"  # assuming that the exe is in the same directory as this script
    
    for language in Languages_To_Test: # Start looping through all languages and their corresponding font
        lang = language[0]
        font = language[1]

        language_arg = "-file=" + lang       # First argument
        font_arg = "-font=" + font           # Second argument
        print(language_arg)
        print(font_arg)
        
        LoopThroughScreens(executable_path, language_arg, font_arg)

def main():
       
    LoopThroughLanguages()

if __name__ == "__main__":
    main()
