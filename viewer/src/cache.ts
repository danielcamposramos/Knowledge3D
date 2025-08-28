export type Store<T> = {
  get(key: string): Promise<T | null>;
  put(key: string, value: T): Promise<void>;
  clear(): Promise<void>;
};

export function openStore<T = unknown>(dbName = 'k3d-cache', storeName = 'k3d'): Store<T> {
  let dbPromise: Promise<IDBDatabase> | null = null;

  function getDB(): Promise<IDBDatabase> {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(dbName, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(storeName)) db.createObjectStore(storeName);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    return dbPromise;
  }

  async function get(key: string): Promise<T | null> {
    try {
      const db = await getDB();
      return await new Promise<T | null>((resolve, reject) => {
        const tx = db.transaction(storeName, 'readonly');
        const st = tx.objectStore(storeName);
        const rq = st.get(key);
        rq.onsuccess = () => resolve((rq.result as T) ?? null);
        rq.onerror = () => reject(rq.error);
      });
    } catch {
      return null;
    }
  }

  async function put(key: string, value: T): Promise<void> {
    try {
      const db = await getDB();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(storeName, 'readwrite');
        const st = tx.objectStore(storeName);
        const rq = st.put(value as any, key);
        rq.onsuccess = () => resolve();
        rq.onerror = () => reject(rq.error);
      });
    } catch {
      // ignore
    }
  }

  async function clear(): Promise<void> {
    try {
      const db = await getDB();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(storeName, 'readwrite');
        const st = tx.objectStore(storeName);
        const rq = st.clear();
        rq.onsuccess = () => resolve();
        rq.onerror = () => reject(rq.error);
      });
    } catch {
      // ignore
    }
  }

  return { get, put, clear };
}

