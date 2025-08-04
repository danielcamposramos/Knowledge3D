export default {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  roots: ['<rootDir>/tests'],
  globals: {
    'ts-jest': {
      tsconfig: './tsconfig.test.json'
    }
  }
};
