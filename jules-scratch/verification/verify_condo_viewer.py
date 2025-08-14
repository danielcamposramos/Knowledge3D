from playwright.sync_api import sync_playwright, expect

def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        try:
            # 1. Navigate to the page
            page.goto("http://localhost:5173")

            # 2. Wait for the dropdown to be populated and select 'math' (default)
            # The dropdown should be populated by the main.ts script
            expect(page.locator("#expert-select")).to_contain_text("math", timeout=10000)

            # Take a screenshot of the default math view
            page.screenshot(path="jules-scratch/verification/verification_math.png")

            # 3. Select the 'physics' expert
            page.select_option("#expert-select", "physics")

            # 4. Give it a moment for the new points to load and render
            # In a real test, we might wait for a specific element or network response.
            # For verification, a short, fixed wait is acceptable.
            page.wait_for_timeout(1000)

            # 5. Take the final screenshot of the physics view
            page.screenshot(path="jules-scratch/verification/verification.png")

            print("Successfully created verification screenshots.")

        except Exception as e:
            print(f"An error occurred during verification: {e}")
            page.screenshot(path="jules-scratch/verification/error.png")

        finally:
            browser.close()

if __name__ == "__main__":
    run_verification()
