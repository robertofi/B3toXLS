from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import re


def get_company_info(symbol):
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ensure the browser runs in headless mode
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--start-maximized')

    # Setup Chrome webdriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                              options=chrome_options)

    try:
        # Navigate to the B3 search page
        driver.get("https://www.b3.com.br/pt_br/busca/?query=" + symbol)

        # Start with a 1-second wait and incrementally increase up to 30 seconds
        for wait_time in range(1,31):
            try:
                wait = WebDriverWait(driver,wait_time)
                wait.until(EC.presence_of_element_located((By.ID,'richSnippet')))

                # Extract the CNPJ and company name
                cnpj_element = driver.find_element(By.XPATH,
                                                   '//*[@id="richSnippet"]/div[3]/div/div[1]/div/p')
                company_name_element = driver.find_element(By.XPATH,
                                                           '//*[@id="richSnippet"]/div[3]/div/div[1]/div/h5')

                cnpj = cnpj_element.text.strip().replace("CNPJ: ","")
                company_name = company_name_element.text.strip()

                # Return CNPJ in the correct format
                return re.sub(r'[^\d./-]','',cnpj),company_name
            except Exception as e:
                if wait_time == 30:
                    raise e  # Raise the exception if max wait time is reached

    except Exception as e:
        print(f"An error occurred: {e}")
        return None,None

    finally:
        driver.quit()


if __name__ == '__main__':
    symbol = "PETR4"
    cnpj,company_name = get_company_info(symbol)
    if cnpj and company_name:
        print(f"CNPJ: {cnpj}")
        print(f"Company Name: {company_name}")
    else:
        print(f"Failed to retrieve data for symbol {symbol}")
